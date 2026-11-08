// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_texture_cache.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <atomic>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"
#include "common/slot_vector.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/textures/texture.h"

namespace Metal {

class Device;
class Framebuffer;
class Image;
class ImageView;
class Sampler;

/// Fragment shader of a blit or MSAA copy pass, defined in mtl_texture_cache.mm.
enum class BlitShader : u8;

using Common::SlotVector;
using VideoCommon::ImageId;
using VideoCommon::NUM_RT;
using VideoCommon::Region2D;
using VideoCommon::RenderTargets;
using VideoCore::Surface::PixelFormat;

struct FormatInfo {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    /// Can be used as a render pass attachment.
    bool attachable = false;
    /// Can be written from shaders.
    bool storage = false;
    /// False when Metal has no format matching the guest layout. Such images get a placeholder
    /// texture and their uploads are skipped.
    bool supported = false;
};

/// Translates a guest pixel format to the Metal format used to store it on this device. Formats
/// that the texture cache converts on upload (ASTC/BCn without native support) report the format
/// of the converted data.
[[nodiscard]] FormatInfo SurfaceFormat(const Device& device, PixelFormat format);

/// How guest depth/stencil texels are laid out compared to the Metal storage format.
enum class DepthStencilLayout : u8 {
    /// Same layout, copies are plain blits.
    Native,
    /// D24 in the high 24 bits and S8 in the low 8 bits, stored as Depth32Float_Stencil8.
    D24S8,
    /// S8 in the high 8 bits and D24 in the low 24 bits, stored as Depth32Float_Stencil8.
    S8D24,
    /// D24 in the low 24 bits, stored as Depth32Float.
    X8D24,
    /// 64-bit texels with a float depth followed by S8, stored as Depth32Float_Stencil8.
    D32S8,
};

class TextureCacheRuntime {
    friend Framebuffer;
    friend Image;
    friend ImageView;
    friend Sampler;

public:
    explicit TextureCacheRuntime(const Device& device, Scheduler& scheduler,
                                 StagingBufferPool& staging_pool);
    ~TextureCacheRuntime();

    TextureCacheRuntime(const TextureCacheRuntime&) = delete;
    TextureCacheRuntime& operator=(const TextureCacheRuntime&) = delete;

    void Finish();

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size, bool deferred = false);
    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);
    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void TickFrame() {}

    [[nodiscard]] u64 GetDeviceLocalMemory() const;
    [[nodiscard]] u64 GetDeviceMemoryUsage() const;
    /// The garbage collector budgets image memory only: the device wide allocation size also
    /// counts buffers and staging memory, which would keep it collecting images that are in use.
    [[nodiscard]] bool CanReportMemoryUsage() const noexcept {
        return false;
    }

    /// A draw samples the depth buffer it renders with (soft particles, fog). The texture cache
    /// reports this for every such draw. Sampling reads memory, where the data stored by
    /// earlier passes is correct: the pass only has to end when draws of this pass wrote the
    /// aspect being sampled and that data still sits in tile memory. The rasterizer knows the
    /// aspects, so it resolves the request before the draw.
    void BarrierFeedbackLoop() {
        scheduler.RequestFeedbackLoopCheck();
    }

    /// Clears deferred by the scheduler run before anything reads or writes their textures,
    /// framebuffer changes don't need them flushed.
    void FlushDeferredClear() const noexcept {}

    [[nodiscard]] bool CanUploadMSAA() const noexcept {
        return true;
    }

    void TransitionImageLayout(Image&) const noexcept {}

    void InsertUploadMemoryBarrier() const noexcept {
        // Uploads are recorded before their users in the same queue, Metal orders them.
    }

    void AccelerateImageUpload(Image& image, const StagingBufferRef& map,
                               std::span<const VideoCommon::SwizzleParameters> swizzles,
                               u32 z_start, u32 z_count);

    [[nodiscard]] bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }

    [[nodiscard]] bool HasNativeBgr() const noexcept {
        return true;
    }

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    [[nodiscard]] bool ShouldReinterpret(Image& dst, Image& src);

    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Region2D& dst_region, const Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    [[nodiscard]] const Device& GetDevice() const noexcept {
        return device;
    }

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    /// Records the guest pages of a new image. Never cleared, so the pages that ever held an
    /// image are a superset of those holding one the GPU modified.
    void MarkImagePages(const VideoCommon::ImageBase& image) noexcept;

    /// False when no image was ever created on the pages of the range, then the texture cache
    /// has nothing to flush for a read of it. Lock free: guest threads ask before taking the
    /// cache mutex, which the GPU thread holds for whole draws.
    [[nodiscard]] bool MayHoldImage(DAddr addr, u64 size) const noexcept;

private:
    /// Copies between images through a staging buffer holding the guest layout. Bit-exact for
    /// any pair of formats with the same bytes per block.
    void CopyThroughBuffer(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    /// Logs the layouts of a pair of copied images the first time they are seen.
    void LogImageCopy(const Image& dst, const Image& src,
                      std::span<const VideoCommon::ImageCopy> copies, bool shareable);

    /// Scaled, flipped, resolving and format converting blits, drawn as a textured quad.
    void BlitWithShader(ImageView& dst, ImageView& src, const Region2D& dst_region,
                        const Region2D& src_region, Tegra::Engines::Fermi2D::Filter filter);

    /// Copies between a single sample and an MSAA texture holding the same guest texels, one
    /// sample per texel of the single sample texture.
    void CopyMSAATextures(id<MTLTexture> dst, id<MTLTexture> src, u32 num_samples,
                          std::span<const VideoCommon::ImageCopy> copies, PixelFormat guest_format);

    /// Shader readable texture for intermediate copies (MSAA uploads, self blits), reused
    /// between calls.
    [[nodiscard]] id<MTLTexture> ScratchTexture(MTLPixelFormat format, u32 width, u32 height,
                                                u32 layers, u32 samples = 1);

    /// Reinterprets depth images as 32-bit color images and back with a render pass, so the
    /// depth/stencil conversion doesn't need a CPU round trip. False when unsupported.
    bool ReinterpretDepthOnGpu(Image& dst, Image& src,
                               std::span<const VideoCommon::ImageCopy> copies);

    /// Unpacks guest depth-stencil texels held in buffer into target with a render pass. False
    /// when the GPU path is unavailable, the caller then converts on the CPU.
    bool UploadDepthStencilOnGpu(id<MTLTexture> target, DepthStencilLayout layout,
                                 bool has_stencil, u32 guest_bytes, id<MTLBuffer> buffer,
                                 size_t offset, const VideoCommon::BufferImageCopy& copy,
                                 u32 width, u32 height, NSUInteger bytes_per_row,
                                 NSUInteger layer_stride);

    /// Packs depth-stencil texels of source into the guest layout in buffer with a render pass.
    bool DownloadDepthStencilOnGpu(id<MTLTexture> source, DepthStencilLayout layout,
                                   bool has_stencil, u32 guest_bytes, id<MTLBuffer> buffer,
                                   size_t offset, const VideoCommon::BufferImageCopy& copy,
                                   u32 width, u32 height, NSUInteger bytes_per_row,
                                   NSUInteger layer_stride);

    /// Compiles the blit shaders on first use, false when they are unavailable.
    bool InitBlitResources();

    /// Render pipeline drawing shader into a single attachment of the given format.
    [[nodiscard]] id<MTLRenderPipelineState> BlitPipeline(BlitShader shader, MTLPixelFormat format,
                                                          NSUInteger samples);

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;

    id<MTLLibrary> blit_library;
    id<MTLSamplerState> blit_linear_sampler;
    id<MTLSamplerState> blit_nearest_sampler;
    id<MTLDepthStencilState> blit_depth_write;
    id<MTLDepthStencilState> blit_depth_stencil_write;
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;
    std::unordered_set<u64> logged_image_copies;
    std::vector<id<MTLTexture>> scratch_textures;
    /// One bit per device page an image was created on, see MayHoldImage.
    std::vector<std::atomic<u64>> image_pages;
    /// Sparse images map their pages anywhere: once one exists every range may hold an image.
    std::atomic<bool> sparse_images{false};
    /// Tick of the latest image download, Finish() waits for it instead of the whole queue.
    u64 download_tick = 0;
    bool blit_init_failed = false;
};

class Image : public VideoCommon::ImageBase {
    friend ImageView;
    friend TextureCacheRuntime;

public:
    explicit Image(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info,
                   GPUVAddr gpu_addr, VAddr cpu_addr);
    explicit Image(const VideoCommon::NullImageParams&);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void UploadMemory(id<MTLBuffer> buffer, size_t offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);
    void UploadMemory(const StagingBufferRef& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(id<MTLBuffer> buffer, size_t offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);
    void DownloadMemory(std::span<id<MTLBuffer>> buffers, std::span<size_t> offsets,
                        std::span<const VideoCommon::BufferImageCopy> copies);
    void DownloadMemory(const StagingBufferRef& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    [[nodiscard]] id<MTLTexture> Handle() const noexcept {
        return texture;
    }

    [[nodiscard]] MTLPixelFormat Format() const noexcept {
        return format_info.format;
    }

    [[nodiscard]] const FormatInfo& GetFormatInfo() const noexcept {
        return format_info;
    }

    [[nodiscard]] DepthStencilLayout GetDepthStencilLayout() const noexcept {
        return depth_layout;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

    bool ScaleUp([[maybe_unused]] bool ignore = false) {
        return false;
    }

    bool ScaleDown([[maybe_unused]] bool ignore = false) {
        return false;
    }

    u64 allocation_tick{};
    /// Blit encoder generation of the last blit touching the texture, see
    /// Scheduler::BlitEncoderFor.
    u64 blit_generation{};

private:
    struct BlockLayout {
        u32 width = 1;
        u32 height = 1;
        u32 bytes = 4;
    };

    /// Uploads to a single sample scratch texture and spreads its texels to the samples.
    void UploadMSAA(id<MTLBuffer> buffer, size_t offset,
                    std::span<const VideoCommon::BufferImageCopy> copies);
    void EncodeUpload(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                      const VideoCommon::BufferImageCopy& copy);
    void EncodeDownload(id<MTLBuffer> buffer, size_t offset,
                        const VideoCommon::BufferImageCopy& copy);
    void UploadDepthStencil(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                            const VideoCommon::BufferImageCopy& copy);
    void DownloadDepthStencil(id<MTLBuffer> buffer, size_t offset,
                              const VideoCommon::BufferImageCopy& copy);
    /// Fallbacks converting on the CPU when the render pass conversions are unavailable.
    void UploadDepthStencilCpu(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                               const VideoCommon::BufferImageCopy& copy);
    void DownloadDepthStencilCpu(id<MTLBuffer> buffer, size_t offset,
                                 const VideoCommon::BufferImageCopy& copy);

    /// View of the texture with another pixel format of the same size, nil when Metal can't
    /// alias them. One view is cached: copies between aliased images reinterpret one way.
    [[nodiscard]] id<MTLTexture> ReinterpretView(MTLPixelFormat format);

    TextureCacheRuntime* runtime{};
    id<MTLTexture> texture;
    id<MTLTexture> reinterpret_view;
    MTLPixelFormat reinterpret_format = MTLPixelFormatInvalid;
    FormatInfo format_info;
    DepthStencilLayout depth_layout = DepthStencilLayout::Native;
    /// Layout of the texels in staging memory, the guest one or the converted one.
    BlockLayout staging_layout;
    bool has_stencil = false;
    bool has_depth = false;
};

class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                       ImageId image_id, Image& image, const SlotVector<Image>& slot_images);
    explicit ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info,
                       const VideoCommon::ImageViewInfo& view_info, GPUVAddr gpu_addr);
    explicit ImageView(TextureCacheRuntime& runtime, const VideoCommon::NullImageViewParams&);
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;
    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// Sampled view for the given shader texture type, nil when unavailable.
    [[nodiscard]] id<MTLTexture> Handle(Shader::TextureType texture_type) const noexcept {
        return views[static_cast<size_t>(texture_type)];
    }

    /// View usable as a render pass attachment: identity swizzle, single level, all layers.
    [[nodiscard]] id<MTLTexture> RenderTarget() const noexcept {
        return render_target;
    }

    /// The texture the views were created from.
    [[nodiscard]] id<MTLTexture> ImageHandle() const noexcept {
        return image_handle;
    }

    [[nodiscard]] MTLPixelFormat Format() const noexcept {
        return view_format;
    }

    [[nodiscard]] u32 Samples() const noexcept {
        return samples;
    }

    /// Shaders sampling this view of a depth-stencil image read its stencil, not its depth.
    [[nodiscard]] bool SamplesStencil() const noexcept {
        return samples_stencil;
    }

    [[nodiscard]] GPUVAddr GpuAddr() const noexcept {
        return gpu_addr;
    }

    [[nodiscard]] u32 BufferSize() const noexcept {
        return buffer_size;
    }

private:
    std::array<id<MTLTexture>, Shader::NUM_TEXTURE_TYPES> views;
    id<MTLTexture> render_target;
    id<MTLTexture> image_handle;
    /// Keeps the placeholder textures of null views alive.
    id<MTLTexture> null_array;
    id<MTLTexture> null_3d;
    MTLPixelFormat view_format = MTLPixelFormatInvalid;
    u32 samples = 1;
    u32 buffer_size = 0;
    bool samples_stencil = false;
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc);

    [[nodiscard]] id<MTLSamplerState> Handle() const noexcept {
        return sampler;
    }

    /// Metal samplers have no LOD bias, shaders must add it to the computed LOD.
    [[nodiscard]] f32 LodBias() const noexcept {
        return lod_bias;
    }

    /// Metal samplers have no custom border colors, the closest predefined one is used.
    [[nodiscard]] bool HasApproximatedBorderColor() const noexcept {
        return approximated_border_color;
    }

private:
    id<MTLSamplerState> sampler;
    f32 lod_bias = 0.0f;
    bool approximated_border_color = false;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime& runtime, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const RenderTargets& key);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&&) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    /// Render pass that loads every attachment and stores the color ones. The depth and stencil
    /// store actions are left unknown: the scheduler finalizes them when the pass ends, with
    /// nothing stored for an aspect the pass only read. Callers copy it before changing the
    /// load actions of a pass.
    [[nodiscard]] MTLRenderPassDescriptor* Descriptor() const noexcept {
        return descriptor;
    }

    [[nodiscard]] u32 Width() const noexcept {
        return width;
    }

    [[nodiscard]] u32 Height() const noexcept {
        return height;
    }

    [[nodiscard]] u32 Samples() const noexcept {
        return samples;
    }

    [[nodiscard]] const std::array<MTLPixelFormat, NUM_RT>& ColorFormats() const noexcept {
        return color_formats;
    }

    [[nodiscard]] MTLPixelFormat DepthFormat() const noexcept {
        return depth_format;
    }

    [[nodiscard]] bool HasStencil() const noexcept {
        return has_stencil;
    }

    /// The texture the depth-stencil attachment is a view of, nil without one.
    [[nodiscard]] id<MTLTexture> DepthImage() const noexcept {
        return depth_image;
    }

private:
    void Setup(std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer);

    MTLRenderPassDescriptor* descriptor;
    id<MTLTexture> depth_image;
    std::array<MTLPixelFormat, NUM_RT> color_formats{};
    MTLPixelFormat depth_format = MTLPixelFormatInvalid;
    u32 width = 0;
    u32 height = 0;
    u32 samples = 1;
    bool has_stencil = false;
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = true;
    static constexpr bool HAS_MSAA_DOWNLOADS = false;

    using Runtime = Metal::TextureCacheRuntime;
    using Image = Metal::Image;
    using ImageAlloc = Metal::ImageAlloc;
    using ImageView = Metal::ImageView;
    using Sampler = Metal::Sampler;
    using Framebuffer = Metal::Framebuffer;
    using AsyncBuffer = Metal::StagingBufferRef;
    using BufferType = id<MTLBuffer>;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace Metal
