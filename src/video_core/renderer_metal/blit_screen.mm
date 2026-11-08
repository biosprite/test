// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_metal/blit_screen.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/textures/decoders.h"

namespace Metal {

namespace {

// TODO: Read the block height from the HLE layer, like the other backends assume 16 GOBs.
constexpr u32 BlockHeightLog2 = 4;
constexpr size_t StagingAlignment = 256;

constexpr const char* PresentShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct PresentVertex {
    float4 position [[position]];
    float2 tex_coord;
};

// Full-viewport quad drawn as a 4 vertex triangle strip. crop is (left, top, right, bottom) in
// normalized texture coordinates, with the origin at the top left like Metal textures.
vertex PresentVertex present_vertex(uint vertex_id [[vertex_id]],
                                    constant float4& crop [[buffer(0)]]) {
    const float2 corner = float2(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u));
    PresentVertex out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.tex_coord = mix(crop.xy, crop.zw, corner);
    return out;
}

fragment float4 present_fragment(PresentVertex in [[stage_in]],
                                 texture2d<float> color_texture [[texture(0)]],
                                 sampler color_sampler [[sampler(0)]]) {
    return color_texture.sample(color_sampler, in.tex_coord);
}
)";

struct GuestFormat {
    MTLPixelFormat pixel_format;
    u32 bytes_per_pixel;
};

std::optional<GuestFormat> ToGuestFormat(Service::android::PixelFormat format,
                                         bool supports_b5g6r5) {
    switch (format) {
    case Service::android::PixelFormat::Rgba8888:
    case Service::android::PixelFormat::Rgbx8888:
        return GuestFormat{MTLPixelFormatRGBA8Unorm, 4};
    case Service::android::PixelFormat::Bgra8888:
        return GuestFormat{MTLPixelFormatBGRA8Unorm, 4};
    case Service::android::PixelFormat::Rgb565:
        if (supports_b5g6r5) {
            return GuestFormat{MTLPixelFormatB5G6R5Unorm, 2};
        }
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

size_t BlendModeIndex(Tegra::BlendMode mode) {
    switch (mode) {
    case Tegra::BlendMode::Opaque:
        return 0;
    case Tegra::BlendMode::Premultiplied:
        return 1;
    case Tegra::BlendMode::Coverage:
        return 2;
    }
    return 0;
}

void ConfigureBlending(MTLRenderPipelineColorAttachmentDescriptor* attachment, size_t mode) {
    attachment.blendingEnabled = mode != 0;
    attachment.rgbBlendOperation = MTLBlendOperationAdd;
    attachment.alphaBlendOperation = MTLBlendOperationAdd;
    attachment.sourceRGBBlendFactor =
        mode == 1 ? MTLBlendFactorOne : MTLBlendFactorSourceAlpha;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorZero;
}

std::string DescribeError(NSError* error) {
    if (!error || !error.localizedDescription) {
        return "unknown error";
    }
    return std::string{error.localizedDescription.UTF8String};
}

} // Anonymous namespace

BlitScreen::BlitScreen(const Device& device_, Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device{device_.GetMTLDevice()}, device_memory{device_memory_},
      supports_b5g6r5{device_.GetFeatures().b5g6r5_format} {
    CreatePipelines();

    for (StagingSlot& slot : staging_slots) {
        // Created at zero and signaled once: libdispatch aborts when a semaphore is released
        // with a value lower than its initial one.
        slot.available = dispatch_semaphore_create(0);
        dispatch_semaphore_signal(slot.available);
    }
}

BlitScreen::~BlitScreen() = default;

void BlitScreen::CreatePipelines() {
    std::string error_message;
    @autoreleasepool {
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:PresentShaderSource]
                                 options:nil
                                   error:&error];
        if (!library) {
            error_message = "Failed to compile present shaders: " + DescribeError(error);
        } else {
            MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
            descriptor.vertexFunction = [library newFunctionWithName:@"present_vertex"];
            descriptor.fragmentFunction = [library newFunctionWithName:@"present_fragment"];
            descriptor.colorAttachments[0].pixelFormat = OutputFormat;

            for (size_t mode = 0; mode < NumBlendModes && error_message.empty(); ++mode) {
                ConfigureBlending(descriptor.colorAttachments[0], mode);
                pipelines[mode] = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                         error:&error];
                if (!pipelines[mode]) {
                    error_message = "Failed to create present pipeline: " + DescribeError(error);
                }
            }
        }

        MTLSamplerDescriptor* sampler = [[MTLSamplerDescriptor alloc] init];
        sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.minFilter = MTLSamplerMinMagFilterLinear;
        sampler.magFilter = MTLSamplerMinMagFilterLinear;
        linear_sampler = [device newSamplerStateWithDescriptor:sampler];
        sampler.minFilter = MTLSamplerMinMagFilterNearest;
        sampler.magFilter = MTLSamplerMinMagFilterNearest;
        nearest_sampler = [device newSamplerStateWithDescriptor:sampler];
        if (error_message.empty() && (!linear_sampler || !nearest_sampler)) {
            error_message = "Failed to create present samplers";
        }
    }
    // Thrown outside of the autorelease pool so that the pool is always drained.
    if (!error_message.empty()) {
        throw std::runtime_error{fmt::format("Metal: {}", error_message)};
    }
}

BlitScreen::StagingSlot& BlitScreen::AcquireStagingSlot(id<MTLCommandBuffer> command_buffer,
                                                        size_t required_size) {
    StagingSlot& slot = staging_slots[next_staging_slot];
    next_staging_slot = (next_staging_slot + 1) % NumStagingSlots;

    // Waits for the GPU to be done with the previous contents of this slot.
    dispatch_semaphore_wait(slot.available, DISPATCH_TIME_FOREVER);
    dispatch_semaphore_t available = slot.available;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      dispatch_semaphore_signal(available);
    }];

    if (slot.capacity < required_size) {
        slot.buffer = [device newBufferWithLength:required_size
                                          options:MTLResourceStorageModeShared];
        slot.capacity = slot.buffer ? required_size : 0;
        slot.buffer.label = @"Eden present staging";
    }
    return slot;
}

id<MTLTexture> BlitScreen::GetLayerTexture(size_t index, u32 width, u32 height,
                                           MTLPixelFormat format) {
    if (layer_textures.size() <= index) {
        layer_textures.resize(index + 1);
    }
    LayerTexture& entry = layer_textures[index];
    if (entry.texture && entry.width == width && entry.height == height &&
        entry.format == format) {
        return entry.texture;
    }
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    // The previous texture stays alive while in-flight command buffers still reference it.
    entry.texture = [device newTextureWithDescriptor:descriptor];
    entry.width = width;
    entry.height = height;
    entry.format = format;
    if (!entry.texture) {
        LOG_ERROR(Render_Metal, "Failed to allocate a {}x{} layer texture", width, height);
    }
    return entry.texture;
}

const u8* BlitScreen::ReadGuestFramebuffer(DAddr address, size_t size) {
    if (const u8* const contiguous = device_memory.GetSpan(address, size)) {
        return contiguous;
    }
    // The framebuffer is not backed by contiguous host memory, gather it page by page.
    guest_scratch.resize(size);
    device_memory.ReadBlockUnsafe(address, guest_scratch.data(), size);
    return guest_scratch.data();
}

void BlitScreen::Draw(id<MTLCommandBuffer> command_buffer, id<MTLTexture> target,
                      std::span<const Tegra::FramebufferConfig> layers,
                      const Layout::FramebufferLayout& layout,
                      RasterizerMetal* rasterizer) {
    ASSERT(target.pixelFormat == OutputFormat);

    struct DisplayLayer {
        id<MTLTexture> texture;
        size_t layer_index;
    };
    std::vector<DisplayLayer> display_layers;
    display_layers.reserve(layers.size());

    pending_uploads.clear();
    size_t staging_size = 0;
    for (size_t index = 0; index < layers.size(); ++index) {
        const Tegra::FramebufferConfig& framebuffer = layers[index];
        if (framebuffer.width == 0 || framebuffer.height == 0) {
            continue;
        }

        if (rasterizer) {
            const auto accel = rasterizer->AccelerateDisplay(
                framebuffer, framebuffer.address + framebuffer.offset, framebuffer.stride);
            if (accel && accel->texture) {
                display_layers.push_back(DisplayLayer{
                    .texture = accel->texture,
                    .layer_index = index,
                });
                continue;
            }
        }

        const auto format = ToGuestFormat(framebuffer.pixel_format, supports_b5g6r5);
        if (!format) {
            UNIMPLEMENTED_MSG("Unsupported framebuffer pixel format: {}",
                              static_cast<u32>(framebuffer.pixel_format));
            continue;
        }
        id<MTLTexture> texture = GetLayerTexture(index, framebuffer.width, framebuffer.height,
                                                 format->pixel_format);
        if (!texture) {
            continue;
        }
        const u32 bytes_per_row = framebuffer.width * format->bytes_per_pixel;
        staging_size = Common::AlignUp(staging_size, StagingAlignment);
        pending_uploads.push_back(PendingUpload{
            .texture = texture,
            .layer_index = index,
            .offset = staging_size,
            .size = static_cast<size_t>(bytes_per_row) * framebuffer.height,
            .bytes_per_row = bytes_per_row,
            .bytes_per_pixel = format->bytes_per_pixel,
        });
        display_layers.push_back(DisplayLayer{
            .texture = texture,
            .layer_index = index,
        });
        staging_size += pending_uploads.back().size;
    }

    if (!pending_uploads.empty()) {
        StagingSlot& slot = AcquireStagingSlot(command_buffer, staging_size);
        if (!slot.buffer) {
            LOG_ERROR(Render_Metal, "Failed to allocate {} bytes of staging memory", staging_size);
            pending_uploads.clear();
        } else {
            u8* const staging = static_cast<u8*>(slot.buffer.contents);
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            blit.label = @"Upload guest framebuffers";
            for (const PendingUpload& upload : pending_uploads) {
                const Tegra::FramebufferConfig& framebuffer = layers[upload.layer_index];
                const DAddr address = framebuffer.address + framebuffer.offset;
                const size_t tiled_size = Tegra::Texture::CalculateSize(
                    true, upload.bytes_per_pixel, framebuffer.stride, framebuffer.height, 1,
                    BlockHeightLog2, 0);
                const u8* const guest = ReadGuestFramebuffer(address, tiled_size);
                Tegra::Texture::UnswizzleTexture(std::span(staging + upload.offset, upload.size),
                                                 std::span(guest, tiled_size),
                                                 upload.bytes_per_pixel, framebuffer.width,
                                                 framebuffer.height, 1, BlockHeightLog2, 0);
                [blit copyFromBuffer:slot.buffer
                           sourceOffset:upload.offset
                      sourceBytesPerRow:upload.bytes_per_row
                    sourceBytesPerImage:0
                             sourceSize:MTLSizeMake(framebuffer.width, framebuffer.height, 1)
                              toTexture:upload.texture
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            [blit endEncoding];
        }
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    encoder.label = @"Composite guest layers";

    const auto& screen = layout.screen;
    if (screen.GetWidth() > 0 && screen.GetHeight() > 0 && !display_layers.empty()) {
        const MTLViewport viewport{
            .originX = static_cast<double>(screen.left),
            .originY = static_cast<double>(screen.top),
            .width = static_cast<double>(screen.GetWidth()),
            .height = static_cast<double>(screen.GetHeight()),
            .znear = 0.0,
            .zfar = 1.0,
        };
        [encoder setViewport:viewport];
        const bool nearest = Settings::values.scaling_filter.GetValue() ==
                             Settings::ScalingFilter::NearestNeighbor;
        [encoder setFragmentSamplerState:nearest ? nearest_sampler : linear_sampler atIndex:0];

        for (const DisplayLayer& layer : display_layers) {
            const Tegra::FramebufferConfig& framebuffer = layers[layer.layer_index];
            const auto crop =
                Tegra::NormalizeCrop(framebuffer, framebuffer.width, framebuffer.height);
            const std::array<f32, 4> crop_rect{crop.left, crop.top, crop.right, crop.bottom};

            [encoder setRenderPipelineState:pipelines[BlendModeIndex(framebuffer.blending)]];
            [encoder setVertexBytes:crop_rect.data() length:sizeof(crop_rect) atIndex:0];
            [encoder setFragmentTexture:layer.texture atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
    }
    [encoder endEncoding];

    // Do not keep layer textures alive through the upload list until the next frame.
    pending_uploads.clear();
}

void BlitScreen::Present(id<MTLCommandBuffer> command_buffer, id<MTLTexture> source,
                         id<MTLTexture> target) const {
    ASSERT(target.pixelFormat == OutputFormat);
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    encoder.label = @"Present frame";
    // The frame was composited at the window size, the drawable only differs during a resize.
    const bool same_size = source.width == target.width && source.height == target.height;
    const std::array<f32, 4> crop_rect{0.0f, 0.0f, 1.0f, 1.0f};
    [encoder setRenderPipelineState:pipelines[BlendModeIndex(Tegra::BlendMode::Opaque)]];
    [encoder setVertexBytes:crop_rect.data() length:sizeof(crop_rect) atIndex:0];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:same_size ? nearest_sampler : linear_sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [encoder endEncoding];
}

} // namespace Metal
