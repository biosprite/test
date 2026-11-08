// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_rasterizer.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/fence_manager.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace Shader {
struct Info;
}

namespace Tegra {
class GPU;
}

namespace Metal {

class Device;
class ShaderFrontend;
struct GraphicsProgram;
struct ComputeProgram;

namespace Dirty {
/// Metal specific dirty flags, after the common ones. They are coarse on purpose: each gates
/// the recomputation of one block of per-draw state, which otherwise costs a few microseconds
/// of register decoding for every draw.
enum : u8 {
    First = VideoCommon::Dirty::LastCommonEntry,
    /// Registers baked into the pipeline key: vertex input, blending, color masks, alpha to
    /// coverage, rasterizer discard.
    PipelineState,
    Viewports,
    Scissors,
    /// Cull, winding, fill mode, depth clip, depth bias, blend color.
    RasterState,
    /// Depth and stencil tests, stencil masks and references.
    DepthStencil,
    Last,
};
static_assert(Last <= std::numeric_limits<u8>::max());
} // namespace Dirty

class AccelerateDMA final : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& buffer_operand,
                            const Tegra::DMA::ImageOperand& image_operand);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
};

/// Host fence: the scheduler tick of the command buffer committed when the fence was queued.
class InnerFence : public VideoCommon::FenceBase {
public:
    explicit InnerFence(Scheduler& scheduler, bool is_stubbed);

    void Queue();
    [[nodiscard]] bool IsSignaled() const;
    void Wait();

private:
    Scheduler& scheduler;
    u64 wait_tick = 0;
};
using Fence = std::shared_ptr<InnerFence>;

/// Host queries are not implemented yet, reports are written directly to guest memory.
struct NullQueryCache {
    bool ShouldWaitAsyncFlushes() const noexcept {
        return false;
    }
    bool HasUncommittedFlushes() const noexcept {
        return false;
    }
    void PopAsyncFlushes() noexcept {}
    void CommitAsyncFlushes() noexcept {}
};

struct FenceManagerParams {
    using FenceType = Fence;
    using BufferCacheType = BufferCache;
    using TextureCacheType = TextureCache;
    using QueryCacheType = NullQueryCache;

    static constexpr bool HAS_ASYNC_CHECK = true;
};

class FenceManager final : public VideoCommon::FenceManager<FenceManagerParams> {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
                          TextureCache& texture_cache, BufferCache& buffer_cache,
                          NullQueryCache& query_cache, Scheduler& scheduler);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;

private:
    Scheduler& scheduler;
};

class RasterizerMetal;

struct FramebufferTextureInfo {
    id<MTLTexture> texture;
    u32 width;
    u32 height;
    u32 scaled_width;
    u32 scaled_height;
};

class RasterizerMetal final : public VideoCore::RasterizerInterface,
                              protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerMetal(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu,
                             Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device, Scheduler& scheduler,
                             StagingBufferPool& staging_pool, BufferCache& buffer_cache,
                             TextureCacheRuntime& texture_cache_runtime,
                             TextureCache& texture_cache, PipelineCache& pipeline_cache);
    ~RasterizerMetal() override;

    RasterizerMetal(const RasterizerMetal&) = delete;
    RasterizerMetal& operator=(const RasterizerMetal&) = delete;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    bool HasDrawTransformFeedback() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;

    [[nodiscard]] std::optional<FramebufferTextureInfo> AccelerateDisplay(
        const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride);

    /// Commits whatever the caches recorded, even without draws, so that work on other command
    /// buffers (the composite pass) is ordered after it.
    void FlushPendingWork();

    /// Runs the deferred clears of a texture another command buffer is about to read (the
    /// displayed image).
    void FlushPendingClearsForTexture(id<MTLTexture> texture);

    /// A composite was dropped because every presenter frame was still in flight.
    void NotifyDroppedFrame() noexcept {
        ++stats.dropped_frames;
    }

    /// Counters of the present thread since the last call, logged with the frame statistics.
    void AddPresenterStats(u64 presented, u64 replaced, u64 failed_drawables,
                           u64 drawable_wait_ns) noexcept {
        stats.presented_frames += presented;
        stats.replaced_frames += replaced;
        stats.failed_drawables += failed_drawables;
        stats.drawable_wait_ns += drawable_wait_ns;
    }

    /// Time spent waiting for an older frame's GPU work before recording the next one, and
    /// whether this frame had to wait at all.
    void AddThrottleWait(u64 wait_ns, bool waited) noexcept {
        stats.throttle_wait_ns += wait_ns;
        if (waited) {
            ++stats.throttle_waits;
        }
    }

private:
    using ResourceList = std::vector<__unsafe_unretained id<MTLResource>>;

    /// Argument buffer and residency information of one shader stage.
    struct StageResources {
        id<MTLBuffer> argument_buffer;
        u64 argument_offset = 0;
        id<MTLBuffer> sizes_buffer;
        u64 sizes_offset = 0;
        ResourceList reads;
        ResourceList writes;

        void Reset() {
            argument_buffer = nil;
            argument_offset = 0;
            sizes_buffer = nil;
            sizes_offset = 0;
            reads.clear();
            writes.clear();
        }
    };

    /// Where the buffers of a stage were recorded by the buffer cache runtime.
    struct StageBindingSource {
        size_t stage = 0;
        bool is_compute = false;
        std::span<const id<MTLTexture>> texture_buffers;
    };

    /// Encoder state that is reapplied only when it changes or when a new encoder starts.
    struct RenderEncoderState {
        u64 generation = ~u64{0};
        id<MTLRenderPipelineState> pipeline;
        id<MTLDepthStencilState> depth_stencil;
        /// Dynamic state last encoded by UpdateDynamicStates, only valid with dynamic_valid.
        struct Dynamic {
            std::array<MTLViewport, Tegra::Engines::Maxwell3D::Regs::NumViewports> viewports;
            std::array<MTLScissorRect, Tegra::Engines::Maxwell3D::Regs::NumViewports> scissors;
            size_t viewport_count;
            MTLCullMode cull_mode;
            MTLWinding winding;
            MTLTriangleFillMode fill_mode;
            MTLDepthClipMode clip_mode;
            std::array<f32, 3> depth_bias;
            std::array<f32, 4> blend_color;
            u32 stencil_refs;
        } dynamic{};
        bool dynamic_valid = false;
        /// Per render stage (vertex, fragment): buffers bound at the descriptor set and buffer
        /// size slots, push constants and the resource lists made resident.
        std::array<id<MTLBuffer>, 2> argument_buffers;
        std::array<u64, 2> argument_offsets{};
        std::array<id<MTLBuffer>, 2> sizes_buffers;
        std::array<u64, 2> sizes_offsets{};
        std::array<std::array<u8, 48>, 2> push_constants{};
        std::array<bool, 2> push_constants_valid{};
        std::array<u64, 2> resident_reads{};
        std::array<u64, 2> resident_writes{};
        std::array<AppliedVertexBuffer, VideoCommon::NUM_VERTEX_BUFFERS> vertex_buffers{};
    };

    /// Attachment formats a pipeline was built for, compared to reuse the last pipeline.
    struct FramebufferFormats {
        std::array<MTLPixelFormat, VideoCommon::NUM_RT> color{};
        MTLPixelFormat depth = MTLPixelFormatInvalid;
        u32 samples = 0;
        bool has_stencil = false;

        bool operator==(const FramebufferFormats&) const = default;
    };

    /// State derived from the Maxwell registers, recomputed only when the dirty flags say so.
    struct DerivedState {
        GraphicsPipeline* pipeline = nullptr;
        /// The pointer alone could be reused by a new program after a shader invalidation.
        const GraphicsProgram* program = nullptr;
        u64 vertex_hash = 0;
        u64 fragment_hash = 0;
        /// Key the pipeline was looked up with, compared against the vertex stream registers.
        GraphicsPipelineKey key{};
        FramebufferFormats framebuffer{};
        Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology pipeline_topology{};

        RenderEncoderState::Dynamic dynamic{};
        id<MTLDepthStencilState> depth_stencil;
        Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology dynamic_topology{};
        u32 fb_width = 0;
        u32 fb_height = 0;
        bool has_depth = false;
        bool has_stencil = false;
        bool dynamic_valid = false;
        /// The draw covers no pixels or culls every triangle.
        bool skip_draw = false;
    };

    /// Counters of the periodic statistics log.
    struct FrameStats {
        u64 frames = 0;
        u64 draws = 0;
        u64 dispatches = 0;
        u64 clears = 0;
        /// Clears drawn into the open pass, as they only cover part of the framebuffer.
        u64 draw_clears = 0;
        u64 pass_barriers = 0;
        u64 pipeline_lookups = 0;
        /// Conditional rendering checks that ran without reading the report back.
        u64 skipped_conditions = 0;
        /// Composites not presented because every presenter frame was still in flight.
        u64 dropped_frames = 0;
        /// Presenter counters: frames that reached a drawable, frames replaced in the mailbox
        /// before the present thread took them, drawables the layer never handed out, and the
        /// time spent waiting for drawables.
        u64 presented_frames = 0;
        u64 replaced_frames = 0;
        u64 failed_drawables = 0;
        u64 drawable_wait_ns = 0;
        /// Time the GPU thread waited for an older composite to keep few frames in flight, and
        /// the composites that had to wait because the queue of frames in flight was full.
        u64 throttle_wait_ns = 0;
        u64 throttle_waits = 0;
        /// GPU execution time of the completed command buffers and their start lag after
        /// commit.
        u64 gpu_busy_ns = 0;
        u64 gpu_lag_ns = 0;
        u64 gpu_timed_buffers = 0;
        Scheduler::Stats scheduler{};
        std::chrono::steady_clock::time_point last_report{};
    };

    struct DrawSetup {
        id<MTLRenderCommandEncoder> encoder;
        const Framebuffer* framebuffer = nullptr;
        /// Cache locks of the draw, released once the caches are done with it: guest threads
        /// wait on them for every write to cached memory.
        std::unique_lock<std::recursive_mutex>* buffer_lock = nullptr;
        std::unique_lock<std::recursive_mutex>* texture_lock = nullptr;
    };

    /// Binds every resource of the current graphics state. Returns false when the draw must be
    /// skipped.
    bool PrepareDraw(bool is_indexed, DrawSetup& setup);

    /// Collects image views and samplers of a stage, in descriptor order.
    void CollectStageTextures(const Shader::Info& info, size_t stage, bool is_compute);

    /// Writes the argument buffer of a stage. Views and samplers are consumed from the cursors.
    bool EncodeStageResources(const ShaderModule& module, const Shader::Info& info,
                              const StageBindingSource& source, size_t& view_cursor,
                              size_t& sampler_cursor, StageResources& out);

    [[nodiscard]] id<MTLArgumentEncoder> ArgumentEncoder(const ShaderModule& module);

    /// slot is 0 for the vertex stage and 1 for the fragment stage.
    void BindRenderStage(id<MTLRenderCommandEncoder> encoder, const ShaderModule* module,
                         const StageResources& resources, size_t slot,
                         std::span<const u8> push_constants);

    [[nodiscard]] GraphicsPipelineKey MakeGraphicsPipelineKey(const GraphicsProgram& program,
                                                              const Framebuffer& framebuffer) const;

    /// True when the vertex stream registers still match the strides and instancing of key.
    [[nodiscard]] bool VertexStreamsMatch(const GraphicsPipelineKey& key) const;

    /// Returns false when the draw must be skipped (empty scissor).
    bool UpdateDynamicStates(id<MTLRenderCommandEncoder> encoder, const Framebuffer& framebuffer);

    [[nodiscard]] id<MTLDepthStencilState> DepthStencilState(bool has_depth, bool has_stencil);

    void PushConstantData(const Shader::Info& info, std::array<u8, 48>& out) const;

    /// Draws a full screen triangle that writes the clear values, honoring masks and scissor.
    void ClearWithDraw(const Framebuffer& framebuffer, u32 color_attachment, u8 color_mask,
                       bool clear_depth, bool clear_stencil, MTLScissorRect rect, u32 base_layer,
                       u32 layer_count);

    /// Runs the deferred clears of images the collected image views read or write.
    void FlushPendingClearsForBoundViews();

    /// Ends the open pass when a bound view samples an aspect of its depth buffer that draws
    /// of the pass wrote, see TextureCacheRuntime::BarrierFeedbackLoop.
    void ResolveFeedbackLoop(const Framebuffer& framebuffer);

    [[nodiscard]] id<MTLRenderPipelineState> UtilityPipeline(const Framebuffer& framebuffer,
                                                             u32 kind, u32 attachment, u8 mask);
    [[nodiscard]] id<MTLFunction> UtilityFunction(const std::string& name, u32 attachment,
                                                  u32 kind);

    /// Commits the recorded work every few thousand draws. ends_render_pass tells that the
    /// caller ends the open render pass anyway, so committing doesn't split one.
    void FlushWork(bool ends_render_pass = false);

    /// Ends the open render pass when shaders in it wrote memory, the only hazard Metal can't
    /// order inside a pass on Apple GPUs.
    void EndPassForShaderWrites();

    /// Logs per frame statistics and memory usage every few seconds.
    void ReportStats();

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;
    BufferCache& buffer_cache;
    TextureCacheRuntime& texture_cache_runtime;
    TextureCache& texture_cache;
    PipelineCache& pipeline_cache;

    std::unique_ptr<ShaderFrontend> shader_frontend;
    NullQueryCache query_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;

    std::vector<VideoCommon::ImageViewInOut> image_views;
    std::vector<VideoCommon::SamplerId> samplers;
    std::vector<u32> sizes_scratch;
    std::array<std::vector<id<MTLTexture>>, VideoCommon::NUM_STAGES> stage_texture_buffers;
    std::vector<id<MTLTexture>> compute_texture_buffers;
    std::array<StageResources, 2> render_resources;
    StageResources compute_resources;
    std::vector<u32> index_scratch;

    Common::unordered_map<const ShaderModule*, id<MTLArgumentEncoder>> argument_encoders;
    Common::unordered_map<u64, id<MTLDepthStencilState>> depth_stencil_states;
    Common::unordered_map<u64, id<MTLRenderPipelineState>> utility_pipelines;
    Common::unordered_map<u64, id<MTLFunction>> utility_functions;
    id<MTLLibrary> utility_library;
    id<MTLSamplerState> fallback_sampler;

    RenderEncoderState render_state;
    DerivedState derived;
    FrameStats stats;
    const GraphicsProgram* current_program = nullptr;
    GraphicsPipeline* current_pipeline = nullptr;
    /// Encoder generation of the last render pass with draws whose shaders write memory.
    u64 pass_writes_generation = ~u64{0};
    NSUInteger max_viewports = 1;
    u32 draw_counter = 0;
    /// Reads of cached memory asking the texture cache what to flush, counted from the guest
    /// threads making them, and how many of them had to take its mutex.
    std::atomic<u64> flush_area_lookups{};
    std::atomic<u64> flush_area_locks{};
};

} // namespace Metal
