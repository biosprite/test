// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/blit_screen.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_swapchain.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/renderer_metal/renderer_metal.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace Metal {

namespace {

id<MTLTexture> CreateRenderTarget(id<MTLDevice> device, u32 width, u32 height) {
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:BlitScreen::OutputFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:descriptor];
}

/// Copies a BGRA8 texture into a CPU visible buffer and waits for the GPU to finish.
bool ReadbackTexture(id<MTLCommandBuffer> command_buffer, id<MTLTexture> texture,
                     id<MTLBuffer> buffer, u32 width, u32 height) {
    constexpr u32 bytes_per_pixel = 4;
    const NSUInteger bytes_per_row = static_cast<NSUInteger>(width) * bytes_per_pixel;

    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit copyFromTexture:texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(width, height, 1)
                        toBuffer:buffer
               destinationOffset:0
          destinationBytesPerRow:bytes_per_row
        destinationBytesPerImage:bytes_per_row * height];
    [blit endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        NSString* const reason = command_buffer.error.localizedDescription;
        LOG_ERROR(Render_Metal, "Readback command buffer failed: {}",
                  reason ? reason.UTF8String : "unknown error");
        return false;
    }
    return true;
}

/// Presents composited frames from its own thread. Acquiring a swapchain drawable blocks until
/// the display releases one, and the GPU thread composites from inside a syncpoint action that
/// holds the syncpoint lock: blocking there delays fence signals and guest wake-ups. The GPU
/// thread renders into a frame texture instead and hands it over; the newest frame replaces a
/// queued one, so presentation never throttles emulation.
class Presenter {
public:
    struct Frame {
        id<MTLTexture> texture;
        /// Value the composite command buffer signals on the composite event once it completed;
        /// the present command buffer waits for it on the GPU.
        u64 composite_value{};
        u32 width{};
        u32 height{};
        /// Queued, being presented, or still read by a present command buffer.
        bool busy{};
    };

    /// Counters of the present thread, read by the GPU thread with TakeStats.
    struct Stats {
        u64 presented = 0;
        /// Frames replaced in the mailbox before the present thread took them.
        u64 replaced = 0;
        u64 failed_drawables = 0;
        u64 drawable_wait_ns = 0;
    };

    explicit Presenter(const Device& device_, Swapchain& swapchain_,
                       const BlitScreen& blit_screen_)
        : device{device_}, swapchain{swapchain_}, blit_screen{blit_screen_},
          state{std::make_shared<State>()} {
        // Presents run on their own queue: command buffers of a queue execute in commit order,
        // and on the cache queue a present committed while the GPU thread already recorded the
        // next frame would wait behind all of that work, holding its drawable for frames.
        present_queue = [device.GetMTLDevice() newCommandQueue];
        present_queue.label = @"Eden present queue";
        // Orders each present after its composite on the GPU, across the two queues, so that
        // this thread never waits for the composite to complete.
        composite_event = [device.GetMTLDevice() newEvent];
        if (!composite_event) {
            throw std::runtime_error{"Failed to create the composite event"};
        }
        composite_event.label = @"Eden composite";
        thread = std::thread([this] { Run(); });
    }

    ~Presenter() {
        {
            std::scoped_lock lock{state->mutex};
            state->stop = true;
        }
        state->cv.notify_all();
        thread.join();
        // The frame textures and the layer must outlive the last present.
        if (last_command_buffer) {
            [last_command_buffer waitUntilCompleted];
        }
    }

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    /// Returns a frame texture of the given size to composite into, or nullptr when every frame
    /// is still queued or being presented, in which case this composite is dropped.
    Frame* AcquireFrame(u32 width, u32 height) {
        std::scoped_lock lock{state->mutex};
        for (Frame& frame : state->frames) {
            if (frame.busy) {
                continue;
            }
            if (!frame.texture || frame.width != width || frame.height != height) {
                frame.texture = CreateRenderTarget(device.GetMTLDevice(), width, height);
                frame.width = width;
                frame.height = height;
                if (!frame.texture) {
                    LOG_ERROR(Render_Metal, "Failed to allocate a {}x{} frame texture", width,
                              height);
                    return nullptr;
                }
            }
            frame.busy = true;
            return &frame;
        }
        return nullptr;
    }

    /// Encodes, at the end of the composite command buffer of frame, the signal its present
    /// waits for. Called after the last encoder of the buffer ended and before it is committed.
    void EncodeComposite(Frame* frame, id<MTLCommandBuffer> composite) {
        frame->composite_value = ++composite_generation;
        [composite encodeSignalEvent:composite_event value:frame->composite_value];
    }

    /// Queues a frame whose composite command buffer has been committed. A frame still waiting
    /// to be presented is replaced, like a mailbox swapchain.
    void Submit(Frame* frame) {
        {
            std::scoped_lock lock{state->mutex};
            if (state->pending) {
                state->pending->busy = false;
                replaced.fetch_add(1, std::memory_order_relaxed);
            }
            state->pending = frame;
        }
        state->cv.notify_one();
    }

    /// Returns the counters accumulated since the last call.
    Stats TakeStats() noexcept {
        return Stats{
            .presented = presented.exchange(0, std::memory_order_relaxed),
            .replaced = replaced.exchange(0, std::memory_order_relaxed),
            .failed_drawables = failed_drawables.exchange(0, std::memory_order_relaxed),
            .drawable_wait_ns = drawable_wait_ns.exchange(0, std::memory_order_relaxed),
        };
    }

private:
    /// Shared with the completion handlers, which may outlive the presenter.
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        /// One being composited, one pending, one taken by the present thread and one per
        /// drawable in flight: a present command buffer holds its frame until its composite
        /// completed on the GPU and the drawable has been blitted.
        std::array<Frame, Swapchain::MaxDrawableCount + 3> frames;
        Frame* pending{};
        bool stop{};
    };

    void Run() {
        Common::SetCurrentThreadName("MetalPresent");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        while (true) {
            Frame* frame;
            {
                std::unique_lock lock{state->mutex};
                state->cv.wait(lock, [this] { return state->stop || state->pending != nullptr; });
                if (state->stop) {
                    if (state->pending) {
                        state->pending->busy = false;
                    }
                    return;
                }
                frame = state->pending;
                state->pending = nullptr;
            }
            Present(*frame);
        }
    }

    void Present(Frame& frame) {
        @autoreleasepool {
            const auto drawable_start = std::chrono::steady_clock::now();
            swapchain.Resize(frame.width, frame.height);
            id<CAMetalDrawable> drawable = swapchain.AcquireNextDrawable();
            drawable_wait_ns.fetch_add(
                Nanoseconds(drawable_start, std::chrono::steady_clock::now()),
                std::memory_order_relaxed);
            if (!drawable) {
                failed_drawables.fetch_add(1, std::memory_order_relaxed);
                Release(state, &frame);
                return;
            }
            id<MTLCommandBuffer> command_buffer = [present_queue commandBuffer];
            command_buffer.label = @"Eden present";
            // The composite runs on the cache queue: the present queue waits for its signal on
            // the GPU, before the encoder reading the frame.
            [command_buffer encodeWaitForEvent:composite_event value:frame.composite_value];
            blit_screen.Present(command_buffer, frame.texture, drawable.texture);
            swapchain.Present(command_buffer, drawable);

            const std::shared_ptr<State> handler_state = state;
            Frame* const released = &frame;
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
              (void)completed;
              Release(handler_state, released);
            }];
            [command_buffer commit];
            last_command_buffer = command_buffer;
            presented.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static u64 Nanoseconds(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) noexcept {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    static void Release(const std::shared_ptr<State>& state, Frame* frame) {
        std::scoped_lock lock{state->mutex};
        frame->busy = false;
    }

    const Device& device;
    Swapchain& swapchain;
    const BlitScreen& blit_screen;
    std::shared_ptr<State> state;
    id<MTLCommandQueue> present_queue;
    id<MTLEvent> composite_event;
    /// Last value signaled on composite_event, only touched by the thread compositing.
    u64 composite_generation{};
    std::atomic<u64> presented{};
    std::atomic<u64> replaced{};
    std::atomic<u64> failed_drawables{};
    std::atomic<u64> drawable_wait_ns{};
    std::thread thread;
    /// Only touched by the present thread and, after it was joined, by the destructor.
    id<MTLCommandBuffer> last_command_buffer;
};

} // Anonymous namespace

struct RendererMetal::Impl {
    explicit Impl(Core::Frontend::EmuWindow& emu_window,
                  Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu)
        : scheduler{device}, staging_pool{device, scheduler},
          buffer_cache_runtime{device, scheduler, staging_pool, device_memory},
          buffer_cache{device_memory, buffer_cache_runtime},
          texture_cache_runtime{device, scheduler, staging_pool},
          texture_cache{texture_cache_runtime, device_memory}, pipeline_cache{device},
          rasterizer{emu_window,    gpu,           device_memory,         device,
                     scheduler,     staging_pool,  buffer_cache,          texture_cache_runtime,
                     texture_cache, pipeline_cache},
          swapchain{device, emu_window.GetWindowInfo()}, blit_screen{device, device_memory},
          presenter{std::make_unique<Presenter>(device, swapchain, blit_screen)} {}

    ~Impl() {
        // Stops presenting before the swapchain and the frame textures go away.
        presenter.reset();
        WaitIdle();
    }

    /// Writes framebuffers rendered by the GPU back to guest memory, where BlitScreen reads them.
    void FlushDisplayedFramebuffers(std::span<const Tegra::FramebufferConfig> framebuffers);

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void WaitIdle() {
        // The queue executes in submission order, so the last command buffer completes last.
        if (last_command_buffer) {
            [last_command_buffer waitUntilCompleted];
            last_command_buffer = nil;
        }
        in_flight_composites.clear();
    }

    /// Keeps the GPU from falling far behind the GPU thread. Nothing else holds the thread
    /// back when the GPU is the bottleneck: the guest ran seconds ahead, every buffer region
    /// stayed in use, presented frames were seconds old. Blocking here stalls the vsync
    /// thread inside its composite, which paces the guest to the GPU. The bound is a queue
    /// depth, not a time: the composites still executing are the frames the GPU lags behind,
    /// so the latency it allows scales with the frame time. A time bound (100 ms once) made
    /// every frame of a GPU bound scene at 10-15 FPS wait for the oldest composite, which
    /// serialized recording and execution into single frames. A game paced by its CPU whose
    /// GPU work completes within a frame or two never queues MaxFramesInFlight composites and
    /// never waits here; when the queue is full only the oldest composite is waited for.
    void ThrottleFrames(id<MTLCommandBuffer> composite) {
        static constexpr size_t MaxFramesInFlight = 3;
        const auto start = std::chrono::steady_clock::now();
        in_flight_composites.push_back(InFlightComposite{.buffer = composite, .committed = start});
        bool waited = false;
        while (!in_flight_composites.empty()) {
            const InFlightComposite& oldest = in_flight_composites.front();
            const MTLCommandBufferStatus status = oldest.buffer.status;
            if (status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError) {
                in_flight_composites.pop_front();
                continue;
            }
            if (in_flight_composites.size() <= MaxFramesInFlight) {
                break;
            }
            [oldest.buffer waitUntilCompleted];
            in_flight_composites.pop_front();
            waited = true;
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        rasterizer.AddThrottleWait(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            waited);
    }

    id<MTLCommandBuffer> NewCommandBuffer(NSString* label) {
        id<MTLCommandBuffer> command_buffer = [device.GetCommandQueue() commandBuffer];
        command_buffer.label = label;
        return command_buffer;
    }

    void RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers,
                          VideoCore::RendererSettings& settings);
    void RenderAppletCaptureLayer(id<MTLCommandBuffer> command_buffer,
                                  std::span<const Tegra::FramebufferConfig> framebuffers);

    // Declaration order matters: users are destroyed before the objects they reference.
    Device device;
    Scheduler scheduler;
    StagingBufferPool staging_pool;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    PipelineCache pipeline_cache;
    RasterizerMetal rasterizer;
    Swapchain swapchain;
    BlitScreen blit_screen;
    std::unique_ptr<Presenter> presenter;

    id<MTLCommandBuffer> last_command_buffer;
    id<MTLTexture> applet_texture;
    /// Composite command buffers of the frames still executing, oldest first.
    struct InFlightComposite {
        id<MTLCommandBuffer> buffer;
        std::chrono::steady_clock::time_point committed;
    };
    std::deque<InFlightComposite> in_flight_composites;

    std::vector<Tegra::FramebufferConfig> applet_capture_layers;
    std::vector<Tegra::FramebufferConfig> screenshot_layer_scratch;
};

void RendererMetal::Impl::FlushDisplayedFramebuffers(
    std::span<const Tegra::FramebufferConfig> framebuffers) {
    for (const Tegra::FramebufferConfig& framebuffer : framebuffers) {
        const DAddr address = framebuffer.address + framebuffer.offset;
        if (const auto info = rasterizer.AccelerateDisplay(framebuffer, address, framebuffer.stride)) {
            // The composite samples the image on another command buffer: a clear still
            // deferred on it has to be recorded before the submission below.
            rasterizer.FlushPendingClearsForTexture(info->texture);
            continue;
        }
        const auto format =
            VideoCore::Surface::PixelFormatFromGPUPixelFormat(framebuffer.pixel_format);
        const u64 size = static_cast<u64>(framebuffer.stride) * framebuffer.height *
                         VideoCore::Surface::BytesPerBlock(format);
        if (address != 0 && size != 0) {
            rasterizer.FlushRegion(address, size, VideoCommon::CacheType::TextureCache);
        }
    }
    // Commands recorded by the caches run before the composite pass on the shared queue.
    rasterizer.FlushPendingWork();
}

void RendererMetal::Impl::RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers,
                                           VideoCore::RendererSettings& settings) {
    if (!settings.screenshot_requested) {
        return;
    }
    const auto layers = Tegra::FilterLayerStack(framebuffers, settings.screenshot_layer_stack,
                                                screenshot_layer_scratch);
    const Layout::FramebufferLayout& layout = settings.screenshot_framebuffer_layout;
    const size_t size = static_cast<size_t>(layout.width) * layout.height * 4;

    bool success = false;
    if (size != 0) {
        id<MTLTexture> target = CreateRenderTarget(device.GetMTLDevice(), layout.width,
                                                   layout.height);
        id<MTLBuffer> readback = [device.GetMTLDevice() newBufferWithLength:size
                                                                   options:MTLResourceStorageModeShared];
        if (target && readback) {
            id<MTLCommandBuffer> command_buffer = NewCommandBuffer(@"Eden screenshot");
            blit_screen.Draw(command_buffer, target, layers, layout, &rasterizer);
            success = ReadbackTexture(command_buffer, target, readback, layout.width,
                                      layout.height);
            last_command_buffer = command_buffer;
            if (success) {
                std::memcpy(settings.screenshot_bits, readback.contents, size);
            }
        }
    }
    if (!success) {
        LOG_ERROR(Render_Metal, "Failed to capture a {}x{} screenshot", layout.width,
                  layout.height);
        if (size != 0) {
            std::memset(settings.screenshot_bits, 0, size);
        }
    }
    settings.screenshot_complete_callback(false);
    settings.screenshot_requested = false;
}

void RendererMetal::Impl::RenderAppletCaptureLayer(
    id<MTLCommandBuffer> command_buffer, std::span<const Tegra::FramebufferConfig> framebuffers) {
    const auto layers = Tegra::FilterLayerStack(
        framebuffers, Service::Nvnflinger::LayerStackId::LastFrame, applet_capture_layers);
    if (layers.empty()) {
        return;
    }
    if (!applet_texture) {
        applet_texture = CreateRenderTarget(device.GetMTLDevice(), VideoCore::Capture::LinearWidth,
                                            VideoCore::Capture::LinearHeight);
        if (!applet_texture) {
            LOG_ERROR(Render_Metal, "Failed to allocate the applet capture texture");
            return;
        }
    }
    blit_screen.Draw(command_buffer, applet_texture, layers, VideoCore::Capture::Layout,
                     &rasterizer);
}

RendererMetal::RendererMetal(Core::Frontend::EmuWindow& emu_window,
                             Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), gpu{gpu_},
      impl{std::make_unique<Impl>(emu_window, device_memory, gpu_)}, rasterizer{&impl->rasterizer} {
    LOG_INFO(Render_Metal, "Metal renderer initialized on {}", impl->device.GetName());
}

RendererMetal::~RendererMetal() = default;

void RendererMetal::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    SCOPE_EXIT {
        render_window.OnFrameDisplayed();
    };

    const bool is_shown = render_window.IsShown();
    bool presented = false;
    @autoreleasepool {
        impl->FlushDisplayedFramebuffers(framebuffers);

        // Screenshots use their own synchronous command buffer, recorded before the frame's one
        // so that BlitScreen never waits on a staging slot held by an uncommitted buffer.
        if (is_shown) {
            impl->RenderScreenshot(framebuffers, renderer_settings);
        }

        id<MTLCommandBuffer> command_buffer = impl->NewCommandBuffer(@"Eden composite");
        impl->RenderAppletCaptureLayer(command_buffer, framebuffers);

        // The frame is composited into a texture of the presenter, which acquires the drawable
        // and presents from its own thread so this call never blocks on the display.
        Presenter::Frame* frame = nullptr;
        if (is_shown) {
            const Layout::FramebufferLayout layout = render_window.GetFramebufferLayout();
            if (layout.width != 0 && layout.height != 0) {
                frame = impl->presenter->AcquireFrame(layout.width, layout.height);
            }
            if (frame) {
                impl->blit_screen.Draw(command_buffer, frame->texture, framebuffers, layout,
                                        &impl->rasterizer);
            }
        }

        if (frame) {
            impl->presenter->EncodeComposite(frame, command_buffer);
        }
        [command_buffer commit];
        impl->last_command_buffer = command_buffer;
        if (frame) {
            impl->presenter->Submit(frame);
            presented = true;
        }
        impl->ThrottleFrames(command_buffer);
    }

    if (!is_shown) {
        return;
    }
    // The frame counter only sees frames that reach the screen.
    if (presented) {
        gpu.RendererFrameEndNotify();
    } else {
        impl->rasterizer.NotifyDroppedFrame();
    }
    const Presenter::Stats present_stats = impl->presenter->TakeStats();
    impl->rasterizer.AddPresenterStats(present_stats.presented, present_stats.replaced,
                                       present_stats.failed_drawables,
                                       present_stats.drawable_wait_ns);
    impl->rasterizer.TickFrame();
}

std::vector<u8> RendererMetal::GetAppletCaptureBuffer() {
    using namespace VideoCore::Capture;

    std::vector<u8> out(TiledSize);
    if (!impl->applet_texture) {
        return out;
    }

    @autoreleasepool {
        constexpr size_t linear_size = static_cast<size_t>(LinearWidth) * LinearHeight *
                                       BytesPerPixel;
        id<MTLBuffer> readback =
            [impl->device.GetMTLDevice() newBufferWithLength:linear_size
                                                    options:MTLResourceStorageModeShared];
        if (!readback) {
            LOG_ERROR(Render_Metal, "Failed to allocate the applet capture readback buffer");
            return out;
        }
        id<MTLCommandBuffer> command_buffer = impl->NewCommandBuffer(@"Eden applet capture");
        const bool success =
            ReadbackTexture(command_buffer, impl->applet_texture, readback, LinearWidth,
                            LinearHeight);
        impl->last_command_buffer = command_buffer;
        if (success) {
            const std::span<const u8> linear(static_cast<const u8*>(readback.contents),
                                             linear_size);
            Tegra::Texture::SwizzleTexture(out, linear, BytesPerPixel, LinearWidth, LinearHeight,
                                           LinearDepth, BlockHeight, BlockDepth);
        }
    }
    return out;
}

} // namespace Metal
