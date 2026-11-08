// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "blit_screen.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <span>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/framebuffer_config.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace Metal {

class Device;
class RasterizerMetal;

/// Composites the guest layers into a BGRA8 render target (swapchain drawable or capture).
class BlitScreen {
public:
    static constexpr MTLPixelFormat OutputFormat = MTLPixelFormatBGRA8Unorm;

    /// @throws std::runtime_error when the present shaders or pipelines cannot be created.
    explicit BlitScreen(const Device& device, Tegra::MaxwellDeviceMemoryManager& device_memory);
    ~BlitScreen();

    BlitScreen(const BlitScreen&) = delete;
    BlitScreen& operator=(const BlitScreen&) = delete;

    /// Uploads the layers from guest memory and encodes their composition into target.
    /// Must be called inside an autorelease pool. Every call reserves a staging slot until the
    /// command buffer completes, so the command buffer must be committed and fewer than
    /// NumStagingSlots calls may be pending on uncommitted command buffers at any time.
    void Draw(id<MTLCommandBuffer> command_buffer, id<MTLTexture> target,
              std::span<const Tegra::FramebufferConfig> layers,
              const Layout::FramebufferLayout& layout,
              RasterizerMetal* rasterizer = nullptr);

    /// Encodes a full screen copy of a composited frame into target (a swapchain drawable).
    /// Only reads immutable state, so it may run on the present thread. Must be called inside
    /// an autorelease pool.
    void Present(id<MTLCommandBuffer> command_buffer, id<MTLTexture> source,
                 id<MTLTexture> target) const;

private:
    static constexpr size_t NumStagingSlots = 4;
    static constexpr size_t NumBlendModes = 3;

    struct StagingSlot {
        id<MTLBuffer> buffer;
        size_t capacity{};
        dispatch_semaphore_t available;
    };

    struct LayerTexture {
        id<MTLTexture> texture;
        u32 width{};
        u32 height{};
        MTLPixelFormat format{MTLPixelFormatInvalid};
    };

    struct PendingUpload {
        id<MTLTexture> texture;
        size_t layer_index{};
        size_t offset{};
        size_t size{};
        u32 bytes_per_row{};
        u32 bytes_per_pixel{};
    };

    void CreatePipelines();
    StagingSlot& AcquireStagingSlot(id<MTLCommandBuffer> command_buffer, size_t required_size);
    id<MTLTexture> GetLayerTexture(size_t index, u32 width, u32 height, MTLPixelFormat format);
    const u8* ReadGuestFramebuffer(DAddr address, size_t size);

    id<MTLDevice> device;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    bool supports_b5g6r5{};

    std::array<id<MTLRenderPipelineState>, NumBlendModes> pipelines;
    id<MTLSamplerState> linear_sampler;
    id<MTLSamplerState> nearest_sampler;

    std::array<StagingSlot, NumStagingSlots> staging_slots;
    size_t next_staging_slot{};

    std::vector<LayerTexture> layer_textures;
    std::vector<PendingUpload> pending_uploads;
    std::vector<u8> guest_scratch;
};

} // namespace Metal
