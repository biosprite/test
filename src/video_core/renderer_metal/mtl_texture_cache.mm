// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <utility>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/util.h"

namespace Metal {

enum class BlitShader : u8 {
    SampleFloat,
    MsFloat,
    ReadUint,
    MsUint,
    ReadSint,
    MsSint,
    Depth,
    DepthMs,
    MsaaToSingleFloat,
    MsaaToSingleUint,
    MsaaToSingleSint,
    MsaaToSingleDepth,
    SingleToMsaaFloat,
    SingleToMsaaUint,
    SingleToMsaaSint,
    SingleToMsaaDepth,
    MsaaToSingleDepthStencil,
    SingleToMsaaDepthStencil,
    DepthToColor,
    DepthStencilToColor,
    ColorToDepth,
    ColorToDepthStencil,
};

namespace {

using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureMipmapFilter;
using VideoCommon::BufferImageCopy;
using VideoCommon::ImageCopy;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::DefaultBlockHeight;
using VideoCore::Surface::DefaultBlockWidth;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::IsPixelFormatASTC;
using VideoCore::Surface::IsPixelFormatBCn;
using VideoCore::Surface::IsPixelFormatSRGB;
using VideoCore::Surface::SurfaceType;

constexpr u32 USAGE_ATTACHABLE = 1 << 0;
constexpr u32 USAGE_STORAGE = 1 << 1;

constexpr f64 UNORM24_MAX = 16777215.0;

/// The image page set has one bit per device page of the whole device address space.
constexpr u64 IMAGE_PAGE_BITS = Core::DEVICE_PAGEBITS;
constexpr size_t NUM_IMAGE_PAGE_WORDS =
    (size_t{1} << (Tegra::MaxwellDeviceTraits::device_virtual_bits - IMAGE_PAGE_BITS)) / 64;

/// Calls func(word, mask) for every word of the image page set the range touches, with the
/// bits of its pages set in mask, until func returns true. True when func did, or when the
/// range leaves the address space the set covers.
template <typename Func>
bool VisitImagePageWords(size_t num_words, DAddr addr, u64 size, Func&& func) {
    const u64 last = (addr + (std::max)(size, u64{1}) - 1) >> IMAGE_PAGE_BITS;
    for (u64 page = addr >> IMAGE_PAGE_BITS; page <= last;) {
        const u64 word = page / 64;
        if (word >= num_words) {
            return true;
        }
        const u64 first_bit = page % 64;
        const u64 count = (std::min)(last - page + 1, 64 - first_bit);
        const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << first_bit;
        if (func(static_cast<size_t>(word), mask)) {
            return true;
        }
        page += count;
    }
    return false;
}

/// Metal requirements on top of a Metal 3 capable device.
enum class Requirement : u8 {
    None,
    /// Packed 16-bit and ETC2/EAC formats are only available on Apple GPUs.
    AppleGpu,
    Astc,
    Bc,
};

void WarnOnce(PixelFormat format, const char* message) {
    static std::array<bool, VideoCore::Surface::MaxPixelFormat> warned{};
    const size_t index = static_cast<size_t>(format);
    if (index < warned.size() && !warned[index]) {
        warned[index] = true;
        LOG_WARNING(Render_Metal, "{}: {}", message, format);
    }
}

bool HasNativeAstc(const Device& device) {
    // ASTC LDR is supported from the A8 (MTLGPUFamilyApple2) onwards.
    return device.GetFeatures().apple_gpu_family >= 2;
}

bool NeedsConversion(const Device& device, PixelFormat format) {
    return (IsPixelFormatASTC(format) && !HasNativeAstc(device)) ||
           (IsPixelFormatBCn(format) && !device.GetFeatures().bc_texture_compression);
}

FormatInfo ConvertedFormat(const Device& device, PixelFormat format) {
    const bool is_srgb = IsPixelFormatSRGB(format);
    const auto rgba8 = [is_srgb] {
        return is_srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
    };
    if (IsPixelFormatASTC(format)) {
        // Must match what VideoCommon::ConvertImage produces for the setting.
        const bool has_bc = device.GetFeatures().bc_texture_compression;
        switch (Settings::values.astc_recompression.GetValue()) {
        case Settings::AstcRecompression::Uncompressed:
            return {rgba8(), false, !is_srgb, true};
        case Settings::AstcRecompression::Bc1:
            return {is_srgb ? MTLPixelFormatBC1_RGBA_sRGB : MTLPixelFormatBC1_RGBA, false, false,
                    has_bc};
        case Settings::AstcRecompression::Bc3:
            return {is_srgb ? MTLPixelFormatBC3_RGBA_sRGB : MTLPixelFormatBC3_RGBA, false, false,
                    has_bc};
        }
        return {rgba8(), false, false, true};
    }
    switch (format) {
    case PixelFormat::BC4_SNORM:
        return {MTLPixelFormatR8Snorm, false, false, true};
    case PixelFormat::BC4_UNORM:
        return {MTLPixelFormatR8Unorm, false, false, true};
    case PixelFormat::BC5_SNORM:
        return {MTLPixelFormatRG8Snorm, false, false, true};
    case PixelFormat::BC5_UNORM:
        return {MTLPixelFormatRG8Unorm, false, false, true};
    case PixelFormat::BC6H_SFLOAT:
    case PixelFormat::BC6H_UFLOAT:
        return {MTLPixelFormatRGBA16Float, false, false, true};
    default:
        return {rgba8(), false, false, true};
    }
}

/// Block layout of the data VideoCommon::ConvertImage leaves in staging memory.
struct HostBlock {
    u32 width;
    u32 height;
    u32 bytes;
};

HostBlock ConvertedBlock(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB:
        return {4, 4, 8};
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB:
        return {4, 4, 16};
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Snorm:
        return {1, 1, 1};
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Snorm:
        return {1, 1, 2};
    case MTLPixelFormatRGBA16Float:
        return {1, 1, 8};
    default:
        return {1, 1, 4};
    }
}

DepthStencilLayout DepthLayoutOf(PixelFormat format) {
    switch (format) {
    case PixelFormat::D24_UNORM_S8_UINT:
        return DepthStencilLayout::D24S8;
    case PixelFormat::S8_UINT_D24_UNORM:
        return DepthStencilLayout::S8D24;
    case PixelFormat::X8_D24_UNORM:
        return DepthStencilLayout::X8D24;
    case PixelFormat::D32_FLOAT_S8_UINT:
        return DepthStencilLayout::D32S8;
    default:
        return DepthStencilLayout::Native;
    }
}

bool HasDepthAspect(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth16Unorm:
    case MTLPixelFormatDepth32Float:
    case MTLPixelFormatDepth24Unorm_Stencil8:
    case MTLPixelFormatDepth32Float_Stencil8:
        return true;
    default:
        return false;
    }
}

bool HasStencilAspect(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatStencil8:
    case MTLPixelFormatDepth24Unorm_Stencil8:
    case MTLPixelFormatDepth32Float_Stencil8:
    case MTLPixelFormatX32_Stencil8:
    case MTLPixelFormatX24_Stencil8:
        return true;
    default:
        return false;
    }
}

/// Format of a view exposing only the stencil of a depth-stencil texture to shaders.
MTLPixelFormat StencilViewFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth32Float_Stencil8:
        return MTLPixelFormatX32_Stencil8;
    case MTLPixelFormatDepth24Unorm_Stencil8:
        return MTLPixelFormatX24_Stencil8;
    case MTLPixelFormatStencil8:
        return MTLPixelFormatStencil8;
    default:
        return MTLPixelFormatInvalid;
    }
}

/// Returns the linear counterpart of sRGB formats, used to validate view reinterpretation.
MTLPixelFormat LinearFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Unorm_sRGB:
        return MTLPixelFormatR8Unorm;
    case MTLPixelFormatRG8Unorm_sRGB:
        return MTLPixelFormatRG8Unorm;
    case MTLPixelFormatRGBA8Unorm_sRGB:
        return MTLPixelFormatRGBA8Unorm;
    case MTLPixelFormatBGRA8Unorm_sRGB:
        return MTLPixelFormatBGRA8Unorm;
    case MTLPixelFormatBC1_RGBA_sRGB:
        return MTLPixelFormatBC1_RGBA;
    case MTLPixelFormatBC2_RGBA_sRGB:
        return MTLPixelFormatBC2_RGBA;
    case MTLPixelFormatBC3_RGBA_sRGB:
        return MTLPixelFormatBC3_RGBA;
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        return MTLPixelFormatBC7_RGBAUnorm;
    case MTLPixelFormatETC2_RGB8_sRGB:
        return MTLPixelFormatETC2_RGB8;
    case MTLPixelFormatETC2_RGB8A1_sRGB:
        return MTLPixelFormatETC2_RGB8A1;
    case MTLPixelFormatEAC_RGBA8_sRGB:
        return MTLPixelFormatEAC_RGBA8;
    case MTLPixelFormatASTC_4x4_sRGB:
        return MTLPixelFormatASTC_4x4_LDR;
    case MTLPixelFormatASTC_5x4_sRGB:
        return MTLPixelFormatASTC_5x4_LDR;
    case MTLPixelFormatASTC_5x5_sRGB:
        return MTLPixelFormatASTC_5x5_LDR;
    case MTLPixelFormatASTC_6x5_sRGB:
        return MTLPixelFormatASTC_6x5_LDR;
    case MTLPixelFormatASTC_6x6_sRGB:
        return MTLPixelFormatASTC_6x6_LDR;
    case MTLPixelFormatASTC_8x5_sRGB:
        return MTLPixelFormatASTC_8x5_LDR;
    case MTLPixelFormatASTC_8x6_sRGB:
        return MTLPixelFormatASTC_8x6_LDR;
    case MTLPixelFormatASTC_8x8_sRGB:
        return MTLPixelFormatASTC_8x8_LDR;
    case MTLPixelFormatASTC_10x5_sRGB:
        return MTLPixelFormatASTC_10x5_LDR;
    case MTLPixelFormatASTC_10x6_sRGB:
        return MTLPixelFormatASTC_10x6_LDR;
    case MTLPixelFormatASTC_10x8_sRGB:
        return MTLPixelFormatASTC_10x8_LDR;
    case MTLPixelFormatASTC_10x10_sRGB:
        return MTLPixelFormatASTC_10x10_LDR;
    case MTLPixelFormatASTC_12x10_sRGB:
        return MTLPixelFormatASTC_12x10_LDR;
    case MTLPixelFormatASTC_12x12_sRGB:
        return MTLPixelFormatASTC_12x12_LDR;
    default:
        return format;
    }
}

/// Metal only allows views between formats of the same size class, and between the sRGB and
/// linear variants of a compressed format.
bool IsViewCompatible(PixelFormat image_format, MTLPixelFormat image_mtl, PixelFormat view_format,
                      MTLPixelFormat view_mtl) {
    if (image_mtl == view_mtl) {
        return true;
    }
    if (GetFormatType(image_format) != SurfaceType::ColorTexture ||
        GetFormatType(view_format) != SurfaceType::ColorTexture) {
        return false;
    }
    const bool image_compressed = DefaultBlockWidth(image_format) != 1;
    const bool view_compressed = DefaultBlockWidth(view_format) != 1;
    if (image_compressed || view_compressed) {
        return LinearFormat(image_mtl) == LinearFormat(view_mtl);
    }
    return BytesPerBlock(image_format) == BytesPerBlock(view_format);
}

SwizzleSource SwapBlueRed(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::R:
        return SwizzleSource::B;
    case SwizzleSource::B:
        return SwizzleSource::R;
    default:
        return value;
    }
}

SwizzleSource SwapSpecial(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::A:
        return SwizzleSource::R;
    case SwizzleSource::R:
        return SwizzleSource::A;
    case SwizzleSource::G:
        return SwizzleSource::B;
    case SwizzleSource::B:
        return SwizzleSource::G;
    default:
        return value;
    }
}

/// Packed formats whose Metal component order differs from the guest one are fixed up with a
/// view swizzle. Metal names packed formats from the least significant bit, guest formats from
/// the most significant one.
void TransformSwizzle(PixelFormat format, std::array<SwizzleSource, 4>& swizzle) {
    switch (format) {
    case PixelFormat::A1B5G5R5_UNORM:
    case PixelFormat::B5G6R5_UNORM:
        std::ranges::transform(swizzle, swizzle.begin(), SwapBlueRed);
        break;
    case PixelFormat::A5B5G5R1_UNORM:
    case PixelFormat::A4B4G4R4_UNORM:
        std::ranges::transform(swizzle, swizzle.begin(), SwapSpecial);
        break;
    default:
        break;
    }
}

MTLTextureSwizzle ToMetalSwizzle(SwizzleSource source) {
    switch (source) {
    case SwizzleSource::Zero:
        return MTLTextureSwizzleZero;
    case SwizzleSource::R:
        return MTLTextureSwizzleRed;
    case SwizzleSource::G:
        return MTLTextureSwizzleGreen;
    case SwizzleSource::B:
        return MTLTextureSwizzleBlue;
    case SwizzleSource::A:
        return MTLTextureSwizzleAlpha;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return MTLTextureSwizzleOne;
    }
    return MTLTextureSwizzleZero;
}

MTLSamplerAddressMode WrapMode(Tegra::Texture::WrapMode mode,
                               Tegra::Texture::TextureFilter filter) {
    using Tegra::Texture::WrapMode;
    switch (mode) {
    case WrapMode::Wrap:
        return MTLSamplerAddressModeRepeat;
    case WrapMode::Mirror:
        return MTLSamplerAddressModeMirrorRepeat;
    case WrapMode::ClampToEdge:
        return MTLSamplerAddressModeClampToEdge;
    case WrapMode::Border:
        return MTLSamplerAddressModeClampToBorderColor;
    case WrapMode::Clamp:
        // GL_CLAMP blends with the border when filtering linearly.
        return filter == Tegra::Texture::TextureFilter::Linear
                   ? MTLSamplerAddressModeClampToBorderColor
                   : MTLSamplerAddressModeClampToEdge;
    case WrapMode::MirrorOnceClampToEdge:
    case WrapMode::MirrorOnceBorder:
    case WrapMode::MirrorOnceClampOGL:
        return MTLSamplerAddressModeMirrorClampToEdge;
    }
    UNIMPLEMENTED_MSG("Unimplemented wrap mode={}", mode);
    return MTLSamplerAddressModeRepeat;
}

MTLCompareFunction CompareFunction(Tegra::Texture::DepthCompareFunc func) {
    using Tegra::Texture::DepthCompareFunc;
    switch (func) {
    case DepthCompareFunc::Never:
        return MTLCompareFunctionNever;
    case DepthCompareFunc::Less:
        return MTLCompareFunctionLess;
    case DepthCompareFunc::Equal:
        return MTLCompareFunctionEqual;
    case DepthCompareFunc::LessEqual:
        return MTLCompareFunctionLessEqual;
    case DepthCompareFunc::Greater:
        return MTLCompareFunctionGreater;
    case DepthCompareFunc::NotEqual:
        return MTLCompareFunctionNotEqual;
    case DepthCompareFunc::GreaterEqual:
        return MTLCompareFunctionGreaterEqual;
    case DepthCompareFunc::Always:
        return MTLCompareFunctionAlways;
    }
    return MTLCompareFunctionNever;
}

/// Picks the closest of the three border colors Metal supports.
MTLSamplerBorderColor BorderColor(const std::array<f32, 4>& color, bool& approximated) {
    const bool transparent = color[3] < 0.5f;
    const f32 luminance = (color[0] + color[1] + color[2]) / 3.0f;
    MTLSamplerBorderColor result;
    std::array<f32, 4> chosen;
    if (transparent) {
        result = MTLSamplerBorderColorTransparentBlack;
        chosen = {0.0f, 0.0f, 0.0f, 0.0f};
    } else if (luminance >= 0.5f) {
        result = MTLSamplerBorderColorOpaqueWhite;
        chosen = {1.0f, 1.0f, 1.0f, 1.0f};
    } else {
        result = MTLSamplerBorderColorOpaqueBlack;
        chosen = {0.0f, 0.0f, 0.0f, 1.0f};
    }
    approximated = color != chosen;
    return result;
}

u32 LevelExtent(NSUInteger base, s32 level) {
    return static_cast<u32>((std::max<NSUInteger>)(base >> level, 1));
}

bool IsLayered(id<MTLTexture> texture) {
    switch (texture.textureType) {
    case MTLTextureType1DArray:
    case MTLTextureType2DArray:
    case MTLTextureTypeCubeArray:
    case MTLTextureType2DMultisampleArray:
        return true;
    default:
        return false;
    }
}

u32 ToUnorm24(f32 depth) {
    const f64 clamped = std::clamp(static_cast<f64>(depth), 0.0, 1.0);
    return static_cast<u32>(std::lround(clamped * UNORM24_MAX));
}

constexpr const char* BlitShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
    float4 src_rect;
    int2 dst_offset;
    int2 src_offset;
    int2 scale;
    int2 src_size;
};

struct BlitVertex {
    float4 position [[position]];
    float2 src_coord;
};

struct DepthOutput {
    float depth [[depth(any)]];
};

// Viewport sized quad drawn as a 4 vertex triangle strip. src_rect holds the source texel
// coordinates of the top left and bottom right corners, reversed for flipped blits.
vertex BlitVertex blit_vertex(uint vertex_id [[vertex_id]],
                              constant BlitParams& params [[buffer(0)]]) {
    const float2 corner = float2(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u));
    BlitVertex out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.src_coord = mix(params.src_rect.xy, params.src_rect.zw, corner);
    return out;
}

static uint2 ClampTexel(int2 coord, int2 size) {
    return uint2(clamp(coord, int2(0), size - 1));
}

static uint2 BlitTexel(float2 coord, constant BlitParams& params) {
    return ClampTexel(int2(floor(coord)), params.src_size);
}

// Single sample texel to the MSAA texel and sample storing it in guest memory.
static uint2 MsaaTexel(float4 position, constant BlitParams& params, thread uint& sample_index) {
    const int2 coord = int2(position.xy) - params.dst_offset + params.src_offset;
    const int2 offset = coord % params.scale;
    sample_index = uint(offset.x + params.scale.x * offset.y);
    return ClampTexel(coord / params.scale, params.src_size);
}

// MSAA texel and sample to the single sample texel storing it in guest memory.
static uint2 SingleTexel(float4 position, uint sample_id, constant BlitParams& params) {
    const int2 offset = int2(int(sample_id) % params.scale.x, int(sample_id) / params.scale.x);
    const int2 coord =
        (int2(position.xy) - params.dst_offset) * params.scale + offset + params.src_offset;
    return ClampTexel(coord, params.src_size);
}

fragment float4 blit_sample_float(BlitVertex in [[stage_in]],
                                  constant BlitParams& params [[buffer(0)]],
                                  texture2d<float> src [[texture(0)]],
                                  sampler src_sampler [[sampler(0)]]) {
    return src.sample(src_sampler, in.src_coord / float2(params.src_size), level(0.0));
}

// Resolves the samples of the source texel.
fragment float4 blit_ms_float(BlitVertex in [[stage_in]],
                              constant BlitParams& params [[buffer(0)]],
                              texture2d_ms<float> src [[texture(0)]]) {
    const uint2 texel = BlitTexel(in.src_coord, params);
    const uint count = src.get_num_samples();
    float4 sum = float4(0.0);
    for (uint i = 0; i < count; ++i) {
        sum += src.read(texel, i);
    }
    return sum / float(count);
}

#define DEFINE_INTEGER_BLITS(NAME, T)                                                          \
    fragment vec<T, 4> blit_read_##NAME(BlitVertex in [[stage_in]],                            \
                                        constant BlitParams& params [[buffer(0)]],             \
                                        texture2d<T> src [[texture(0)]]) {                     \
        return src.read(BlitTexel(in.src_coord, params));                                      \
    }                                                                                          \
    fragment vec<T, 4> blit_ms_##NAME(BlitVertex in [[stage_in]],                              \
                                      constant BlitParams& params [[buffer(0)]],               \
                                      texture2d_ms<T> src [[texture(0)]]) {                    \
        return src.read(BlitTexel(in.src_coord, params), 0);                                   \
    }

#define DEFINE_MSAA_COPIES(NAME, T)                                                            \
    fragment vec<T, 4> msaa_to_single_##NAME(float4 position [[position]],                     \
                                             constant BlitParams& params [[buffer(0)]],        \
                                             texture2d_ms<T> src [[texture(0)]]) {             \
        uint sample_index;                                                                     \
        const uint2 texel = MsaaTexel(position, params, sample_index);                         \
        return src.read(texel, min(sample_index, src.get_num_samples() - 1));                  \
    }                                                                                          \
    fragment vec<T, 4> single_to_msaa_##NAME(float4 position [[position]],                     \
                                             uint sample_id [[sample_id]],                     \
                                             constant BlitParams& params [[buffer(0)]],        \
                                             texture2d<T> src [[texture(0)]]) {                \
        return src.read(SingleTexel(position, sample_id, params));                             \
    }

DEFINE_INTEGER_BLITS(uint, uint)
DEFINE_INTEGER_BLITS(sint, int)
DEFINE_MSAA_COPIES(float, float)
DEFINE_MSAA_COPIES(uint, uint)
DEFINE_MSAA_COPIES(sint, int)

fragment DepthOutput blit_depth(BlitVertex in [[stage_in]],
                                constant BlitParams& params [[buffer(0)]],
                                depth2d<float> src [[texture(0)]]) {
    DepthOutput out;
    out.depth = src.read(BlitTexel(in.src_coord, params));
    return out;
}

fragment DepthOutput blit_depth_ms(BlitVertex in [[stage_in]],
                                   constant BlitParams& params [[buffer(0)]],
                                   depth2d_ms<float> src [[texture(0)]]) {
    DepthOutput out;
    out.depth = src.read(BlitTexel(in.src_coord, params), 0);
    return out;
}

fragment DepthOutput msaa_to_single_depth(float4 position [[position]],
                                          constant BlitParams& params [[buffer(0)]],
                                          depth2d_ms<float> src [[texture(0)]]) {
    uint sample_index;
    const uint2 texel = MsaaTexel(position, params, sample_index);
    DepthOutput out;
    out.depth = src.read(texel, min(sample_index, src.get_num_samples() - 1));
    return out;
}

fragment DepthOutput single_to_msaa_depth(float4 position [[position]],
                                          uint sample_id [[sample_id]],
                                          constant BlitParams& params [[buffer(0)]],
                                          depth2d<float> src [[texture(0)]]) {
    DepthOutput out;
    out.depth = src.read(SingleTexel(position, sample_id, params));
    return out;
}

// Bit reinterpretation between depth images and color images holding the guest bytes, viewed
// as R32Uint (4 byte texels) or RG32Uint (8 byte texels). The layout values match
// DepthStencilLayout: 1 D24S8, 2 S8D24, 3 X8D24, 4 D32S8 (float depth, then S8), 0 a plain
// float depth.
struct DepthStencilOutput {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

static uint2 CopyTexel(float4 position, constant BlitParams& params) {
    return ClampTexel(int2(position.xy) - params.dst_offset + params.src_offset, params.src_size);
}

static uint ToUnorm24(float depth) {
    return uint(saturate(depth) * 16777215.0 + 0.5);
}

static uint2 PackDepthStencil(float depth, uint stencil, uint layout) {
    switch (layout) {
    case 1:
        return uint2((ToUnorm24(depth) << 8) | (stencil & 0xFFu), 0u);
    case 2:
        return uint2(ToUnorm24(depth) | (stencil << 24), 0u);
    case 3:
        return uint2(ToUnorm24(depth), 0u);
    case 4:
        return uint2(as_type<uint>(depth), stencil & 0xFFu);
    default:
        return uint2(as_type<uint>(depth), 0u);
    }
}

static float UnpackDepth(uint2 value, uint layout) {
    switch (layout) {
    case 1:
        return float(value.x >> 8) / 16777215.0;
    case 2:
    case 3:
        return float(value.x & 0xFFFFFFu) / 16777215.0;
    default:
        return as_type<float>(value.x);
    }
}

static uint UnpackStencil(uint2 value, uint layout) {
    switch (layout) {
    case 1:
        return value.x & 0xFFu;
    case 2:
        return value.x >> 24;
    case 4:
        return value.y & 0xFFu;
    default:
        return 0u;
    }
}

fragment uint4 depth_to_color(float4 position [[position]],
                              constant BlitParams& params [[buffer(0)]],
                              constant uint& layout [[buffer(1)]],
                              depth2d<float> depth [[texture(0)]]) {
    const uint2 texel = CopyTexel(position, params);
    return uint4(PackDepthStencil(depth.read(texel), 0u, layout), 0u, 0u);
}

fragment uint4 depth_stencil_to_color(float4 position [[position]],
                                      constant BlitParams& params [[buffer(0)]],
                                      constant uint& layout [[buffer(1)]],
                                      depth2d<float> depth [[texture(0)]],
                                      texture2d<uint> stencil [[texture(1)]]) {
    const uint2 texel = CopyTexel(position, params);
    return uint4(PackDepthStencil(depth.read(texel), stencil.read(texel).r, layout), 0u, 0u);
}

fragment DepthOutput color_to_depth(float4 position [[position]],
                                    constant BlitParams& params [[buffer(0)]],
                                    constant uint& layout [[buffer(1)]],
                                    texture2d<uint> src [[texture(0)]]) {
    DepthOutput out;
    out.depth = UnpackDepth(src.read(CopyTexel(position, params)).rg, layout);
    return out;
}

fragment DepthStencilOutput color_to_depth_stencil(float4 position [[position]],
                                                   constant BlitParams& params [[buffer(0)]],
                                                   constant uint& layout [[buffer(1)]],
                                                   texture2d<uint> src [[texture(0)]]) {
    const uint2 value = src.read(CopyTexel(position, params)).rg;
    DepthStencilOutput out;
    out.depth = UnpackDepth(value, layout);
    out.stencil = UnpackStencil(value, layout);
    return out;
}

// Sample (de)interleaving of depth-stencil images, the stencil is read through an X32_Stencil8
// view bound as a uint texture.
fragment DepthStencilOutput msaa_to_single_depth_stencil(float4 position [[position]],
                                                         constant BlitParams& params [[buffer(0)]],
                                                         depth2d_ms<float> depth [[texture(0)]],
                                                         texture2d_ms<uint> stencil [[texture(1)]]) {
    uint sample_index;
    const uint2 texel = MsaaTexel(position, params, sample_index);
    const uint sample = min(sample_index, depth.get_num_samples() - 1);
    DepthStencilOutput out;
    out.depth = depth.read(texel, sample);
    out.stencil = stencil.read(texel, sample).r;
    return out;
}

fragment DepthStencilOutput single_to_msaa_depth_stencil(float4 position [[position]],
                                                         uint sample_id [[sample_id]],
                                                         constant BlitParams& params [[buffer(0)]],
                                                         depth2d<float> depth [[texture(0)]],
                                                         texture2d<uint> stencil [[texture(1)]]) {
    const uint2 texel = SingleTexel(position, sample_id, params);
    DepthStencilOutput out;
    out.depth = depth.read(texel);
    out.stencil = stencil.read(texel).r;
    return out;
}
)";

/// Indexed by BlitShader.
constexpr std::array BlitShaderNames{
    "blit_sample_float",    "blit_ms_float",          "blit_read_uint",
    "blit_ms_uint",         "blit_read_sint",         "blit_ms_sint",
    "blit_depth",           "blit_depth_ms",          "msaa_to_single_float",
    "msaa_to_single_uint",  "msaa_to_single_sint",    "msaa_to_single_depth",
    "single_to_msaa_float", "single_to_msaa_uint",    "single_to_msaa_sint",
    "single_to_msaa_depth", "msaa_to_single_depth_stencil", "single_to_msaa_depth_stencil",
    "depth_to_color",       "depth_stencil_to_color", "color_to_depth",
    "color_to_depth_stencil",
};
static_assert(BlitShaderNames.size() == static_cast<size_t>(BlitShader::ColorToDepthStencil) + 1);

/// Must match BlitParams in BlitShaderSource.
struct BlitParams {
    std::array<f32, 4> src_rect{};
    std::array<s32, 2> dst_offset{};
    std::array<s32, 2> src_offset{};
    std::array<s32, 2> scale{1, 1};
    std::array<s32, 2> src_size{1, 1};
};
static_assert(sizeof(BlitParams) == 48);

/// Data type shaders read from and write to a texture of a given format.
enum class TexelType : u8 {
    Float,
    Uint,
    Sint,
    Depth,
    Stencil,
};

TexelType TexelTypeOf(MTLPixelFormat format) {
    if (HasDepthAspect(format)) {
        return TexelType::Depth;
    }
    if (HasStencilAspect(format)) {
        return TexelType::Stencil;
    }
    switch (format) {
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA32Uint:
        return TexelType::Uint;
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA32Sint:
        return TexelType::Sint;
    default:
        return TexelType::Float;
    }
}

std::optional<BlitShader> BlitShaderFor(TexelType type, bool msaa_source) {
    switch (type) {
    case TexelType::Float:
        return msaa_source ? BlitShader::MsFloat : BlitShader::SampleFloat;
    case TexelType::Uint:
        return msaa_source ? BlitShader::MsUint : BlitShader::ReadUint;
    case TexelType::Sint:
        return msaa_source ? BlitShader::MsSint : BlitShader::ReadSint;
    case TexelType::Depth:
        return msaa_source ? BlitShader::DepthMs : BlitShader::Depth;
    case TexelType::Stencil:
        break;
    }
    return std::nullopt;
}

std::optional<BlitShader> MsaaCopyShaderFor(TexelType type, bool msaa_source, bool stencil) {
    switch (type) {
    case TexelType::Float:
        return msaa_source ? BlitShader::MsaaToSingleFloat : BlitShader::SingleToMsaaFloat;
    case TexelType::Uint:
        return msaa_source ? BlitShader::MsaaToSingleUint : BlitShader::SingleToMsaaUint;
    case TexelType::Sint:
        return msaa_source ? BlitShader::MsaaToSingleSint : BlitShader::SingleToMsaaSint;
    case TexelType::Depth:
        if (stencil) {
            return msaa_source ? BlitShader::MsaaToSingleDepthStencil
                               : BlitShader::SingleToMsaaDepthStencil;
        }
        return msaa_source ? BlitShader::MsaaToSingleDepth : BlitShader::SingleToMsaaDepth;
    case TexelType::Stencil:
        break;
    }
    return std::nullopt;
}

/// True when rect covers a whole width x height level.
bool CoversLevel(const MTLScissorRect& rect, NSUInteger width, NSUInteger height) {
    return rect.x == 0 && rect.y == 0 && rect.width >= width && rect.height >= height;
}

/// Points the first color attachment, or the depth and stencil attachments, of pass at target.
/// Attachments the pass fully overwrites skip loading their old contents, which saves a read
/// of the whole surface into tile memory on Apple GPUs.
void SetBlitAttachment(MTLRenderPassDescriptor* pass, id<MTLTexture> target, NSUInteger level,
                       NSUInteger slice, bool covers_level, bool writes_stencil = false) {
    const MTLPixelFormat format = target.pixelFormat;
    const auto setup = [&](MTLRenderPassAttachmentDescriptor* attachment, bool overwritten) {
        attachment.texture = target;
        attachment.level = level;
        if (target.textureType == MTLTextureType3D) {
            attachment.depthPlane = slice;
        } else {
            attachment.slice = slice;
        }
        attachment.loadAction = overwritten ? MTLLoadActionDontCare : MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionStore;
    };
    if (HasDepthAspect(format)) {
        setup(pass.depthAttachment, covers_level);
        if (HasStencilAspect(format)) {
            setup(pass.stencilAttachment, covers_level && writes_stencil);
            if (!writes_stencil) {
                // Attached with its depth but untouched by the pass: its memory keeps the
                // contents loaded into tile memory, storing them back would only cost bandwidth.
                pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
            }
        }
    } else {
        setup(pass.colorAttachments[0], covers_level);
    }
}

/// Draws a quad covering rect in its own render pass.
void DrawBlitQuad(Scheduler& scheduler, MTLRenderPassDescriptor* pass,
                  id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_state,
                  id<MTLTexture> source, id<MTLSamplerState> sampler, const BlitParams& params,
                  const MTLScissorRect& rect, id<MTLTexture> second_source = nil,
                  u32 layout = 0) {
    scheduler.EndEncoding(Scheduler::PassEnd::BlitPass);
    id<MTLRenderCommandEncoder> encoder = scheduler.RenderEncoder(pass);
    if (!encoder) {
        return;
    }
    [encoder setRenderPipelineState:pipeline];
    if (depth_state) {
        [encoder setDepthStencilState:depth_state];
    }
    const MTLViewport viewport{
        .originX = static_cast<double>(rect.x),
        .originY = static_cast<double>(rect.y),
        .width = static_cast<double>(rect.width),
        .height = static_cast<double>(rect.height),
        .znear = 0.0,
        .zfar = 1.0,
    };
    [encoder setViewport:viewport];
    [encoder setScissorRect:rect];
    [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
    [encoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [encoder setFragmentTexture:source atIndex:0];
    if (second_source) {
        [encoder setFragmentTexture:second_source atIndex:1];
    }
    [encoder setFragmentBytes:&layout length:sizeof(layout) atIndex:1];
    if (sampler) {
        [encoder setFragmentSamplerState:sampler atIndex:0];
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    // Later commands of the cache must not land in this pass.
    scheduler.EndEncoding(Scheduler::PassEnd::BlitPass);
}

/// Clips the destination span [d0, d1) to [0, limit), moving the source span with it.
bool ClipBlitSpan(s32& d0, s32& d1, f32& s0, f32& s1, s32 limit) {
    const s32 c0 = (std::max)(d0, 0);
    const s32 c1 = (std::min)(d1, limit);
    if (c0 >= c1) {
        return false;
    }
    const f32 ratio = (s1 - s0) / static_cast<f32>(d1 - d0);
    const f32 n0 = s0 + static_cast<f32>(c0 - d0) * ratio;
    const f32 n1 = s0 + static_cast<f32>(c1 - d0) * ratio;
    d0 = c0;
    d1 = c1;
    s0 = n0;
    s1 = n1;
    return true;
}

} // Anonymous namespace

FormatInfo SurfaceFormat(const Device& device, PixelFormat format) {
    if (NeedsConversion(device, format)) {
        return ConvertedFormat(device, format);
    }
    MTLPixelFormat mtl_format = MTLPixelFormatInvalid;
    u32 usage = 0;
    Requirement requirement = Requirement::None;
    const auto set = [&](MTLPixelFormat f, u32 u, Requirement r = Requirement::None) {
        mtl_format = f;
        usage = u;
        requirement = r;
    };
    constexpr u32 AS = USAGE_ATTACHABLE | USAGE_STORAGE;
    constexpr u32 A = USAGE_ATTACHABLE;
    constexpr Requirement APPLE = Requirement::AppleGpu;
    constexpr Requirement ASTC = Requirement::Astc;
    constexpr Requirement BC = Requirement::Bc;
    switch (format) {
    case PixelFormat::A8B8G8R8_UNORM: set(MTLPixelFormatRGBA8Unorm, AS); break;
    case PixelFormat::A8B8G8R8_SNORM: set(MTLPixelFormatRGBA8Snorm, AS); break;
    case PixelFormat::A8B8G8R8_SINT: set(MTLPixelFormatRGBA8Sint, AS); break;
    case PixelFormat::A8B8G8R8_UINT: set(MTLPixelFormatRGBA8Uint, AS); break;
    case PixelFormat::R5G6B5_UNORM: set(MTLPixelFormatB5G6R5Unorm, A, APPLE); break;
    // Formats fixed up with a view swizzle can't be rendered to with the right layout.
    case PixelFormat::B5G6R5_UNORM: set(MTLPixelFormatB5G6R5Unorm, 0, APPLE); break;
    case PixelFormat::A1R5G5B5_UNORM: set(MTLPixelFormatBGR5A1Unorm, A, APPLE); break;
    case PixelFormat::A2B10G10R10_UNORM: set(MTLPixelFormatRGB10A2Unorm, AS); break;
    case PixelFormat::A2B10G10R10_UINT: set(MTLPixelFormatRGB10A2Uint, AS); break;
    case PixelFormat::A2R10G10B10_UNORM: set(MTLPixelFormatBGR10A2Unorm, A); break;
    case PixelFormat::A1B5G5R5_UNORM: set(MTLPixelFormatBGR5A1Unorm, 0, APPLE); break;
    case PixelFormat::A5B5G5R1_UNORM: set(MTLPixelFormatA1BGR5Unorm, 0, APPLE); break;
    case PixelFormat::R8_UNORM: set(MTLPixelFormatR8Unorm, AS); break;
    case PixelFormat::R8_SNORM: set(MTLPixelFormatR8Snorm, AS); break;
    case PixelFormat::R8_SINT: set(MTLPixelFormatR8Sint, AS); break;
    case PixelFormat::R8_UINT: set(MTLPixelFormatR8Uint, AS); break;
    case PixelFormat::R16G16B16A16_FLOAT: set(MTLPixelFormatRGBA16Float, AS); break;
    case PixelFormat::R16G16B16A16_UNORM: set(MTLPixelFormatRGBA16Unorm, AS); break;
    case PixelFormat::R16G16B16A16_SNORM: set(MTLPixelFormatRGBA16Snorm, AS); break;
    case PixelFormat::R16G16B16A16_SINT: set(MTLPixelFormatRGBA16Sint, AS); break;
    case PixelFormat::R16G16B16A16_UINT: set(MTLPixelFormatRGBA16Uint, AS); break;
    case PixelFormat::B10G11R11_FLOAT: set(MTLPixelFormatRG11B10Float, AS); break;
    case PixelFormat::R32G32B32A32_UINT: set(MTLPixelFormatRGBA32Uint, AS); break;
    case PixelFormat::BC1_RGBA_UNORM: set(MTLPixelFormatBC1_RGBA, 0, BC); break;
    case PixelFormat::BC2_UNORM: set(MTLPixelFormatBC2_RGBA, 0, BC); break;
    case PixelFormat::BC3_UNORM: set(MTLPixelFormatBC3_RGBA, 0, BC); break;
    case PixelFormat::BC4_UNORM: set(MTLPixelFormatBC4_RUnorm, 0, BC); break;
    case PixelFormat::BC4_SNORM: set(MTLPixelFormatBC4_RSnorm, 0, BC); break;
    case PixelFormat::BC5_UNORM: set(MTLPixelFormatBC5_RGUnorm, 0, BC); break;
    case PixelFormat::BC5_SNORM: set(MTLPixelFormatBC5_RGSnorm, 0, BC); break;
    case PixelFormat::BC7_UNORM: set(MTLPixelFormatBC7_RGBAUnorm, 0, BC); break;
    case PixelFormat::BC6H_UFLOAT: set(MTLPixelFormatBC6H_RGBUfloat, 0, BC); break;
    case PixelFormat::BC6H_SFLOAT: set(MTLPixelFormatBC6H_RGBFloat, 0, BC); break;
    case PixelFormat::ASTC_2D_4X4_UNORM: set(MTLPixelFormatASTC_4x4_LDR, 0, ASTC); break;
    case PixelFormat::B8G8R8A8_UNORM: set(MTLPixelFormatBGRA8Unorm, A); break;
    case PixelFormat::R32G32B32A32_FLOAT: set(MTLPixelFormatRGBA32Float, AS); break;
    case PixelFormat::R32G32B32A32_SINT: set(MTLPixelFormatRGBA32Sint, AS); break;
    case PixelFormat::R32G32_FLOAT: set(MTLPixelFormatRG32Float, AS); break;
    case PixelFormat::R32G32_SINT: set(MTLPixelFormatRG32Sint, AS); break;
    case PixelFormat::R32_FLOAT: set(MTLPixelFormatR32Float, AS); break;
    case PixelFormat::R16_FLOAT: set(MTLPixelFormatR16Float, AS); break;
    case PixelFormat::R16_UNORM: set(MTLPixelFormatR16Unorm, AS); break;
    case PixelFormat::R16_SNORM: set(MTLPixelFormatR16Snorm, AS); break;
    case PixelFormat::R16_UINT: set(MTLPixelFormatR16Uint, AS); break;
    case PixelFormat::R16_SINT: set(MTLPixelFormatR16Sint, AS); break;
    case PixelFormat::R16G16_UNORM: set(MTLPixelFormatRG16Unorm, AS); break;
    case PixelFormat::R16G16_FLOAT: set(MTLPixelFormatRG16Float, AS); break;
    case PixelFormat::R16G16_UINT: set(MTLPixelFormatRG16Uint, AS); break;
    case PixelFormat::R16G16_SINT: set(MTLPixelFormatRG16Sint, AS); break;
    case PixelFormat::R16G16_SNORM: set(MTLPixelFormatRG16Snorm, AS); break;
    case PixelFormat::A8B8G8R8_SRGB: set(MTLPixelFormatRGBA8Unorm_sRGB, A); break;
    case PixelFormat::R8G8_UNORM: set(MTLPixelFormatRG8Unorm, AS); break;
    case PixelFormat::R8G8_SNORM: set(MTLPixelFormatRG8Snorm, AS); break;
    case PixelFormat::R8G8_SINT: set(MTLPixelFormatRG8Sint, AS); break;
    case PixelFormat::R8G8_UINT: set(MTLPixelFormatRG8Uint, AS); break;
    case PixelFormat::R32G32_UINT: set(MTLPixelFormatRG32Uint, AS); break;
    case PixelFormat::R16G16B16X16_FLOAT: set(MTLPixelFormatRGBA16Float, AS); break;
    case PixelFormat::R32_UINT: set(MTLPixelFormatR32Uint, AS); break;
    case PixelFormat::R32_SINT: set(MTLPixelFormatR32Sint, AS); break;
    case PixelFormat::ASTC_2D_8X8_UNORM: set(MTLPixelFormatASTC_8x8_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_8X5_UNORM: set(MTLPixelFormatASTC_8x5_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_5X4_UNORM: set(MTLPixelFormatASTC_5x4_LDR, 0, ASTC); break;
    case PixelFormat::B8G8R8A8_SRGB: set(MTLPixelFormatBGRA8Unorm_sRGB, A); break;
    case PixelFormat::BC1_RGBA_SRGB: set(MTLPixelFormatBC1_RGBA_sRGB, 0, BC); break;
    case PixelFormat::BC2_SRGB: set(MTLPixelFormatBC2_RGBA_sRGB, 0, BC); break;
    case PixelFormat::BC3_SRGB: set(MTLPixelFormatBC3_RGBA_sRGB, 0, BC); break;
    case PixelFormat::BC7_SRGB: set(MTLPixelFormatBC7_RGBAUnorm_sRGB, 0, BC); break;
    case PixelFormat::A4B4G4R4_UNORM: set(MTLPixelFormatABGR4Unorm, 0, APPLE); break;
    case PixelFormat::ASTC_2D_4X4_SRGB: set(MTLPixelFormatASTC_4x4_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_8X8_SRGB: set(MTLPixelFormatASTC_8x8_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_8X5_SRGB: set(MTLPixelFormatASTC_8x5_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_5X4_SRGB: set(MTLPixelFormatASTC_5x4_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_5X5_UNORM: set(MTLPixelFormatASTC_5x5_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_5X5_SRGB: set(MTLPixelFormatASTC_5x5_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X8_UNORM: set(MTLPixelFormatASTC_10x8_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X8_SRGB: set(MTLPixelFormatASTC_10x8_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_6X6_UNORM: set(MTLPixelFormatASTC_6x6_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_6X6_SRGB: set(MTLPixelFormatASTC_6x6_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X6_UNORM: set(MTLPixelFormatASTC_10x6_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X6_SRGB: set(MTLPixelFormatASTC_10x6_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X5_UNORM: set(MTLPixelFormatASTC_10x5_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X5_SRGB: set(MTLPixelFormatASTC_10x5_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X10_UNORM: set(MTLPixelFormatASTC_10x10_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_10X10_SRGB: set(MTLPixelFormatASTC_10x10_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_12X10_UNORM: set(MTLPixelFormatASTC_12x10_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_12X10_SRGB: set(MTLPixelFormatASTC_12x10_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_12X12_UNORM: set(MTLPixelFormatASTC_12x12_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_12X12_SRGB: set(MTLPixelFormatASTC_12x12_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_8X6_UNORM: set(MTLPixelFormatASTC_8x6_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_8X6_SRGB: set(MTLPixelFormatASTC_8x6_sRGB, 0, ASTC); break;
    case PixelFormat::ASTC_2D_6X5_UNORM: set(MTLPixelFormatASTC_6x5_LDR, 0, ASTC); break;
    case PixelFormat::ASTC_2D_6X5_SRGB: set(MTLPixelFormatASTC_6x5_sRGB, 0, ASTC); break;
    case PixelFormat::E5B9G9R9_FLOAT: set(MTLPixelFormatRGB9E5Float, 0); break;
    case PixelFormat::ETC2_RGB_UNORM: set(MTLPixelFormatETC2_RGB8, 0, APPLE); break;
    case PixelFormat::ETC2_RGBA_UNORM: set(MTLPixelFormatEAC_RGBA8, 0, APPLE); break;
    case PixelFormat::ETC2_RGB_PTA_UNORM: set(MTLPixelFormatETC2_RGB8A1, 0, APPLE); break;
    case PixelFormat::ETC2_RGB_SRGB: set(MTLPixelFormatETC2_RGB8_sRGB, 0, APPLE); break;
    case PixelFormat::ETC2_RGBA_SRGB: set(MTLPixelFormatEAC_RGBA8_sRGB, 0, APPLE); break;
    case PixelFormat::ETC2_RGB_PTA_SRGB: set(MTLPixelFormatETC2_RGB8A1_sRGB, 0, APPLE); break;
    case PixelFormat::EAC_R11_UNORM: set(MTLPixelFormatEAC_R11Unorm, 0, APPLE); break;
    case PixelFormat::EAC_R11_SNORM: set(MTLPixelFormatEAC_R11Snorm, 0, APPLE); break;
    case PixelFormat::EAC_R11G11_UNORM: set(MTLPixelFormatEAC_RG11Unorm, 0, APPLE); break;
    case PixelFormat::EAC_R11G11_SNORM: set(MTLPixelFormatEAC_RG11Snorm, 0, APPLE); break;
    case PixelFormat::D32_FLOAT: set(MTLPixelFormatDepth32Float, A); break;
    case PixelFormat::D16_UNORM: set(MTLPixelFormatDepth16Unorm, A); break;
    case PixelFormat::X8_D24_UNORM: set(MTLPixelFormatDepth32Float, A); break;
    case PixelFormat::S8_UINT: set(MTLPixelFormatStencil8, A); break;
    // Apple GPUs have no 24-bit depth, every D24 variant is converted to a float depth.
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT: set(MTLPixelFormatDepth32Float_Stencil8, A); break;
    // No Metal equivalent: R32G32B32_FLOAT, G4R4_UNORM.
    default: break;
    }
    const DeviceFeatures& features = device.GetFeatures();
    bool supported = mtl_format != MTLPixelFormatInvalid;
    switch (requirement) {
    case Requirement::None:
        break;
    case Requirement::AppleGpu:
        supported &= features.apple_gpu_family != 0;
        break;
    case Requirement::Astc:
        supported &= HasNativeAstc(device);
        break;
    case Requirement::Bc:
        supported &= features.bc_texture_compression;
        break;
    }
    if (!supported) {
        // Placeholder of the same size class, the contents of these images are not uploaded.
        return FormatInfo{
            .format = MTLPixelFormatRGBA8Unorm,
            .attachable = false,
            .storage = false,
            .supported = false,
        };
    }
    return FormatInfo{
        .format = mtl_format,
        .attachable = (usage & USAGE_ATTACHABLE) != 0,
        .storage = (usage & USAGE_STORAGE) != 0,
        .supported = true,
    };
}

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                         StagingBufferPool& staging_pool_)
    : device{device_}, scheduler{scheduler_}, staging_pool{staging_pool_},
      image_pages(NUM_IMAGE_PAGE_WORDS) {}

TextureCacheRuntime::~TextureCacheRuntime() = default;

void TextureCacheRuntime::MarkImagePages(const VideoCommon::ImageBase& image) noexcept {
    if (image.info.is_sparse) {
        // Its pages are the mapped segments of the GPU range, not a contiguous CPU range.
        sparse_images.store(true, std::memory_order_relaxed);
    }
    if (image.guest_size_bytes == 0) {
        return;
    }
    VisitImagePageWords(image_pages.size(), image.cpu_addr, image.guest_size_bytes,
                        [this](size_t word, u64 mask) {
                            image_pages[word].fetch_or(mask, std::memory_order_relaxed);
                            return false;
                        });
}

bool TextureCacheRuntime::MayHoldImage(DAddr addr, u64 size) const noexcept {
    if (sparse_images.load(std::memory_order_relaxed)) {
        return true;
    }
    return VisitImagePageWords(image_pages.size(), addr, size, [this](size_t word, u64 mask) {
        return (image_pages[word].load(std::memory_order_relaxed) & mask) != 0;
    });
}

bool TextureCacheRuntime::InitBlitResources() {
    if (blit_library) {
        return true;
    }
    if (blit_init_failed) {
        return false;
    }
    id<MTLDevice> mtl_device = device.GetMTLDevice();
    @autoreleasepool {
        NSError* error = nil;
        blit_library =
            [mtl_device newLibraryWithSource:[NSString stringWithUTF8String:BlitShaderSource]
                                     options:nil
                                       error:&error];
        if (!blit_library) {
            LOG_ERROR(Render_Metal, "Failed to compile the blit shaders: {}",
                      error ? error.localizedDescription.UTF8String : "unknown error");
            blit_init_failed = true;
            return false;
        }
        MTLSamplerDescriptor* sampler = [[MTLSamplerDescriptor alloc] init];
        sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.minFilter = MTLSamplerMinMagFilterLinear;
        sampler.magFilter = MTLSamplerMinMagFilterLinear;
        blit_linear_sampler = [mtl_device newSamplerStateWithDescriptor:sampler];
        sampler.minFilter = MTLSamplerMinMagFilterNearest;
        sampler.magFilter = MTLSamplerMinMagFilterNearest;
        blit_nearest_sampler = [mtl_device newSamplerStateWithDescriptor:sampler];

        MTLDepthStencilDescriptor* depth = [[MTLDepthStencilDescriptor alloc] init];
        depth.depthCompareFunction = MTLCompareFunctionAlways;
        depth.depthWriteEnabled = YES;
        blit_depth_write = [mtl_device newDepthStencilStateWithDescriptor:depth];

        // Stencil values exported by the shader replace the reference value.
        MTLStencilDescriptor* stencil = [[MTLStencilDescriptor alloc] init];
        stencil.stencilCompareFunction = MTLCompareFunctionAlways;
        stencil.depthStencilPassOperation = MTLStencilOperationReplace;
        stencil.readMask = 0xFF;
        stencil.writeMask = 0xFF;
        depth.frontFaceStencil = stencil;
        depth.backFaceStencil = stencil;
        blit_depth_stencil_write = [mtl_device newDepthStencilStateWithDescriptor:depth];
    }
    return true;
}

id<MTLRenderPipelineState> TextureCacheRuntime::BlitPipeline(BlitShader shader,
                                                             MTLPixelFormat format,
                                                             NSUInteger samples) {
    const u64 key = static_cast<u64>(shader) | (static_cast<u64>(format) << 8) |
                    (static_cast<u64>(samples) << 40);
    if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
        return it->second;
    }
    if (!InitBlitResources()) {
        return nil;
    }
    id<MTLRenderPipelineState> pipeline;
    @autoreleasepool {
        const char* const name = BlitShaderNames[static_cast<size_t>(shader)];
        MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.vertexFunction = [blit_library newFunctionWithName:@"blit_vertex"];
        descriptor.fragmentFunction =
            [blit_library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (HasDepthAspect(format)) {
            descriptor.depthAttachmentPixelFormat = format;
            if (HasStencilAspect(format)) {
                descriptor.stencilAttachmentPixelFormat = format;
            }
        } else {
            descriptor.colorAttachments[0].pixelFormat = format;
        }
        descriptor.rasterSampleCount = samples;
        NSError* error = nil;
        pipeline = [device.GetMTLDevice() newRenderPipelineStateWithDescriptor:descriptor
                                                                         error:&error];
        if (!pipeline) {
            LOG_ERROR(Render_Metal, "Failed to create the {} pipeline for format {}: {}", name,
                      static_cast<u32>(format),
                      error ? error.localizedDescription.UTF8String : "unknown error");
        }
    }
    // Failures are cached too, so they are only reported once.
    blit_pipelines.emplace(key, pipeline);
    return pipeline;
}

void TextureCacheRuntime::Finish() {
    // The texture cache only calls this to read back image downloads, waiting for the last one
    // avoids submitting and draining the queue when nothing was downloaded.
    if (download_tick != 0 && !scheduler.IsFree(download_tick)) {
        scheduler.Wait(download_tick, Scheduler::WaitReason::ImageDownload);
    }
}

StagingBufferRef TextureCacheRuntime::UploadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Upload, deferred);
}

StagingBufferRef TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Download, deferred);
}

void TextureCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    // Image memory is accounted by the texture cache itself. The working set only sizes the
    // garbage collector thresholds: with a smaller value they drop to the defaults, below the
    // working set of large games, and the collector then evicts and synchronously downloads
    // images every frame.
    return device.GetRecommendedMaxWorkingSetSize();
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetMTLDevice().currentAllocatedSize;
}

void TextureCacheRuntime::AccelerateImageUpload(Image& image, const StagingBufferRef&,
                                                std::span<const VideoCommon::SwizzleParameters>,
                                                u32, u32) {
    // Images never get the AcceleratedUpload flag on Metal yet.
    WarnOnce(image.info.format, "Accelerated image uploads are not implemented");
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src, std::span<const ImageCopy> copies) {
    if (!dst.texture || !src.texture) {
        return;
    }
    scheduler.FlushPendingClearsFor({dst.texture, src.texture},
                                    Scheduler::ClearFlushReason::ImageOperation);
    // Statistics only: copies between images of one size and layout could be avoided by
    // backing the images with views of one allocation.
    const bool shareable =
        dst.texture.width == src.texture.width && dst.texture.height == src.texture.height &&
        dst.texture.depth == src.texture.depth &&
        dst.texture.mipmapLevelCount == src.texture.mipmapLevelCount &&
        dst.texture.arrayLength == src.texture.arrayLength &&
        dst.texture.sampleCount == src.texture.sampleCount &&
        dst.texture.textureType == src.texture.textureType &&
        BytesPerBlock(dst.info.format) == BytesPerBlock(src.info.format) &&
        DefaultBlockWidth(dst.info.format) == DefaultBlockWidth(src.info.format) &&
        DefaultBlockHeight(dst.info.format) == DefaultBlockHeight(src.info.format);
    scheduler.CountImageCopy(shareable);
    LogImageCopy(dst, src, copies, shareable);
    if (dst.texture.sampleCount != src.texture.sampleCount || !dst.format_info.supported ||
        !src.format_info.supported) {
        CopyThroughBuffer(dst, src, copies);
        return;
    }
    // Aliased images of different formats are copied through a view of the source in the
    // destination format when the formats alias, which spares the staging round trip (a
    // download and an upload, each a blit encoder) that CopyThroughBuffer needs.
    id<MTLTexture> source = src.texture;
    if (dst.Format() != src.Format()) {
        const bool depth_stencil = HasDepthAspect(src.Format()) || HasStencilAspect(src.Format()) ||
                                   HasDepthAspect(dst.Format()) || HasStencilAspect(dst.Format());
        if (depth_stencil || True(src.flags & ImageFlagBits::Converted) ||
            True(dst.flags & ImageFlagBits::Converted) ||
            !IsViewCompatible(src.info.format, src.Format(), dst.info.format, dst.Format())) {
            CopyThroughBuffer(dst, src, copies);
            return;
        }
        source = src.ReinterpretView(dst.Format());
        if (!source) {
            CopyThroughBuffer(dst, src, copies);
            return;
        }
    }
    id<MTLBlitCommandEncoder> blit = scheduler.BlitEncoderFor(dst.blit_generation, src.blit_generation);
    const bool src_3d = src.texture.textureType == MTLTextureType3D;
    const bool dst_3d = dst.texture.textureType == MTLTextureType3D;
    for (const ImageCopy& copy : copies) {
        const s32 src_level = copy.src_subresource.base_level;
        const s32 dst_level = copy.dst_subresource.base_level;
        const u32 width = (std::min)({copy.extent.width,
                                      LevelExtent(src.texture.width, src_level) -
                                          static_cast<u32>(copy.src_offset.x),
                                      LevelExtent(dst.texture.width, dst_level) -
                                          static_cast<u32>(copy.dst_offset.x)});
        const u32 height = (std::min)({copy.extent.height,
                                       LevelExtent(src.texture.height, src_level) -
                                           static_cast<u32>(copy.src_offset.y),
                                       LevelExtent(dst.texture.height, dst_level) -
                                           static_cast<u32>(copy.dst_offset.y)});
        if (src_3d && dst_3d) {
            [blit copyFromTexture:source
                      sourceSlice:0
                      sourceLevel:src_level
                     sourceOrigin:MTLOriginMake(copy.src_offset.x, copy.src_offset.y,
                                                copy.src_offset.z)
                       sourceSize:MTLSizeMake(width, height, copy.extent.depth)
                        toTexture:dst.texture
                 destinationSlice:0
                 destinationLevel:dst_level
                destinationOrigin:MTLOriginMake(copy.dst_offset.x, copy.dst_offset.y,
                                                copy.dst_offset.z)];
            continue;
        }
        // Copies between 3D slices and array layers are expanded one slice at a time.
        const u32 count =
            src_3d || dst_3d
                ? (std::max)({copy.extent.depth, static_cast<u32>(copy.src_subresource.num_layers),
                              static_cast<u32>(copy.dst_subresource.num_layers)})
                : static_cast<u32>(copy.src_subresource.num_layers);
        for (u32 i = 0; i < count; ++i) {
            const u32 src_slice = src_3d ? 0 : copy.src_subresource.base_layer + i;
            const u32 dst_slice = dst_3d ? 0 : copy.dst_subresource.base_layer + i;
            [blit copyFromTexture:source
                      sourceSlice:src_slice
                      sourceLevel:src_level
                     sourceOrigin:MTLOriginMake(copy.src_offset.x, copy.src_offset.y,
                                                src_3d ? copy.src_offset.z + i : 0)
                       sourceSize:MTLSizeMake(width, height, 1)
                        toTexture:dst.texture
                 destinationSlice:dst_slice
                 destinationLevel:dst_level
                destinationOrigin:MTLOriginMake(copy.dst_offset.x, copy.dst_offset.y,
                                                dst_3d ? copy.dst_offset.z + i : 0)];
        }
    }
}

void TextureCacheRuntime::LogImageCopy(const Image& dst, const Image& src,
                                       std::span<const ImageCopy> copies, bool shareable) {
    // Each distinct pair of image layouts is described once, enough to tell what a game
    // keeps copying between aliased images.
    static constexpr size_t MAX_LOGGED_COPIES = 48;
    if (copies.empty() || logged_image_copies.size() >= MAX_LOGGED_COPIES) {
        return;
    }
    const auto key_of = [](const VideoCommon::ImageInfo& info) {
        return static_cast<u64>(info.size.width) ^ (static_cast<u64>(info.size.height) << 16) ^
               (static_cast<u64>(info.size.depth) << 32) ^ (static_cast<u64>(info.format) << 44) ^
               (static_cast<u64>(info.resources.levels) << 52) ^
               (static_cast<u64>(info.resources.layers) << 36) ^
               (static_cast<u64>(info.type) << 60) ^ (static_cast<u64>(info.num_samples) << 56);
    };
    const u64 key = (key_of(src.info) * 0x9E3779B97F4A7C15ull) ^ key_of(dst.info) ^
                    (static_cast<u64>(copies.size()) << 3);
    if (!logged_image_copies.insert(key).second) {
        return;
    }
    const auto describe = [](const VideoCommon::ImageInfo& info) {
        return fmt::format("{}x{}x{} {} type {} levels {} layers {} samples {}", info.size.width,
                           info.size.height, info.size.depth, info.format,
                           static_cast<int>(info.type), info.resources.levels,
                           info.resources.layers, info.num_samples);
    };
    const ImageCopy& first = copies.front();
    LOG_INFO(Render_Metal,
             "Image copy: [{}] -> [{}], {} copies, first {}x{}x{} level {} layer {} -> level {} "
             "layer {}, shareable {}",
             describe(src.info), describe(dst.info), copies.size(), first.extent.width,
             first.extent.height, first.extent.depth, first.src_subresource.base_level,
             first.src_subresource.base_layer, first.dst_subresource.base_level,
             first.dst_subresource.base_layer, shareable);
}

void TextureCacheRuntime::CopyThroughBuffer(Image& dst, Image& src,
                                            std::span<const ImageCopy> copies) {
    if (!dst.texture || !src.texture) {
        return;
    }
    scheduler.FlushPendingClearsFor({dst.texture, src.texture},
                                    Scheduler::ClearFlushReason::ImageOperation);
    const PixelFormat src_format = src.info.format;
    const PixelFormat dst_format = dst.info.format;
    if (BytesPerBlock(src_format) != BytesPerBlock(dst_format) || !src.format_info.supported ||
        !dst.format_info.supported || True(src.flags & ImageFlagBits::Converted) ||
        True(dst.flags & ImageFlagBits::Converted) || src.texture.sampleCount > 1 ||
        dst.texture.sampleCount > 1) {
        LOG_WARNING(Render_Metal, "Unsupported image copy from {} to {}", src_format, dst_format);
        return;
    }
    const bool needs_cpu = src.depth_layout != DepthStencilLayout::Native ||
                           dst.depth_layout != DepthStencilLayout::Native;
    if (needs_cpu && ReinterpretDepthOnGpu(dst, src, copies)) {
        return;
    }
    const u32 bytes_per_block = BytesPerBlock(src_format);
    const bool src_3d = src.info.type == ImageType::e3D;
    const bool dst_3d = dst.info.type == ImageType::e3D;
    for (const ImageCopy& copy : copies) {
        const u32 blocks_w = Common::DivCeil(copy.extent.width, DefaultBlockWidth(src_format));
        const u32 blocks_h = Common::DivCeil(copy.extent.height, DefaultBlockHeight(src_format));
        const u32 slices = (std::max)({copy.extent.depth,
                                       static_cast<u32>(copy.src_subresource.num_layers),
                                       static_cast<u32>(copy.dst_subresource.num_layers)});
        const size_t bytes =
            static_cast<size_t>(blocks_w) * blocks_h * bytes_per_block * slices;

        const auto make_copy = [&](PixelFormat format, bool is_3d,
                                   const VideoCommon::SubresourceLayers& subresource,
                                   const VideoCommon::Offset3D& offset) {
            VideoCommon::SubresourceLayers layers = subresource;
            layers.num_layers = is_3d ? 1 : static_cast<s32>(slices);
            return BufferImageCopy{
                .buffer_offset = 0,
                .buffer_size = bytes,
                .buffer_row_length = blocks_w * DefaultBlockWidth(format),
                .buffer_image_height = blocks_h * DefaultBlockHeight(format),
                .image_subresource = layers,
                .image_offset = offset,
                .image_extent = {blocks_w * DefaultBlockWidth(format),
                                 blocks_h * DefaultBlockHeight(format), is_3d ? slices : 1},
            };
        };
        const std::array src_copy{
            make_copy(src_format, src_3d, copy.src_subresource, copy.src_offset)};
        const std::array dst_copy{
            make_copy(dst_format, dst_3d, copy.dst_subresource, copy.dst_offset)};

        const StagingBufferRef staging = staging_pool.Request(bytes, MemoryUsage::Download);
        src.DownloadMemory(staging.buffer, staging.offset, src_copy);
        if (needs_cpu) {
            // Depth/stencil layouts are converted on the CPU, the guest bytes must exist first.
            scheduler.Finish();
        } else {
            // The upload reads the staging memory the download writes: separate encoders
            // order them.
            scheduler.EndBlitEncoding();
        }
        dst.UploadMemory(staging.buffer, staging.offset, dst_copy);
    }
}

namespace {

/// Uint format a color texture holding packed depth-stencil bytes can be viewed as, Invalid
/// when Metal has no such view. Packed formats are only reinterpreted on Apple GPUs, other
/// GPUs restrict their views.
MTLPixelFormat PackedUintFormat(MTLPixelFormat format, u32 bytes, bool apple_gpu) {
    switch (format) {
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
        return bytes == 4 ? MTLPixelFormatR32Uint : MTLPixelFormatInvalid;
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
        return bytes == 4 && apple_gpu ? MTLPixelFormatR32Uint : MTLPixelFormatInvalid;
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
        return bytes == 8 ? MTLPixelFormatRG32Uint : MTLPixelFormatInvalid;
    default:
        return MTLPixelFormatInvalid;
    }
}

} // Anonymous namespace

bool TextureCacheRuntime::ReinterpretDepthOnGpu(Image& dst, Image& src,
                                                std::span<const ImageCopy> copies) {
    const bool src_is_depth = HasDepthAspect(src.Format());
    if (src_is_depth == HasDepthAspect(dst.Format())) {
        return false;
    }
    Image& depth = src_is_depth ? src : dst;
    Image& color = src_is_depth ? dst : src;
    const DepthStencilLayout layout = depth.depth_layout;
    const bool known_layout =
        layout != DepthStencilLayout::Native || depth.Format() == MTLPixelFormatDepth32Float;
    const u32 guest_bytes = depth.staging_layout.bytes;
    const bool apple_gpu = device.GetFeatures().apple_gpu_family != 0;
    const MTLPixelFormat packed_format =
        PackedUintFormat(color.Format(), guest_bytes, apple_gpu);
    if (!known_layout || BytesPerBlock(color.info.format) != guest_bytes ||
        DefaultBlockWidth(color.info.format) != 1 || packed_format == MTLPixelFormatInvalid ||
        src.texture.textureType == MTLTextureType3D ||
        dst.texture.textureType == MTLTextureType3D ||
        (dst.texture.usage & MTLTextureUsageRenderTarget) == 0) {
        return false;
    }
    const bool stencil = depth.has_stencil && layout != DepthStencilLayout::X8D24 &&
                         layout != DepthStencilLayout::Native;
    const MTLPixelFormat stencil_format = StencilViewFormat(depth.Format());
    if (stencil && stencil_format == MTLPixelFormatInvalid) {
        return false;
    }
    BlitShader shader;
    MTLPixelFormat target_format;
    if (src_is_depth) {
        shader = stencil ? BlitShader::DepthStencilToColor : BlitShader::DepthToColor;
        target_format = packed_format;
    } else {
        shader = stencil ? BlitShader::ColorToDepthStencil : BlitShader::ColorToDepth;
        target_format = dst.Format();
    }
    id<MTLRenderPipelineState> pipeline = BlitPipeline(shader, target_format, 1);
    if (!pipeline) {
        return false;
    }
    id<MTLDepthStencilState> depth_state = nil;
    if (!src_is_depth) {
        depth_state = stencil ? blit_depth_stencil_write : blit_depth_write;
    }
    const u32 layout_value = static_cast<u32>(layout);

    for (const ImageCopy& copy : copies) {
        const s32 src_level = copy.src_subresource.base_level;
        const s32 dst_level = copy.dst_subresource.base_level;
        const s32 level_width = static_cast<s32>(LevelExtent(dst.texture.width, dst_level));
        const s32 level_height = static_cast<s32>(LevelExtent(dst.texture.height, dst_level));
        const s32 x0 = std::clamp(copy.dst_offset.x, 0, level_width);
        const s32 y0 = std::clamp(copy.dst_offset.y, 0, level_height);
        const s32 x1 = (std::min)(x0 + static_cast<s32>(copy.extent.width), level_width);
        const s32 y1 = (std::min)(y0 + static_cast<s32>(copy.extent.height), level_height);
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }
        const MTLScissorRect rect{
            .x = static_cast<NSUInteger>(x0),
            .y = static_cast<NSUInteger>(y0),
            .width = static_cast<NSUInteger>(x1 - x0),
            .height = static_cast<NSUInteger>(y1 - y0),
        };
        const bool covers_level = CoversLevel(rect, static_cast<NSUInteger>(level_width),
                                              static_cast<NSUInteger>(level_height));
        const BlitParams params{
            .dst_offset = {copy.dst_offset.x, copy.dst_offset.y},
            .src_offset = {copy.src_offset.x, copy.src_offset.y},
            .src_size = {static_cast<s32>(LevelExtent(src.texture.width, src_level)),
                         static_cast<s32>(LevelExtent(src.texture.height, src_level))},
        };
        const s32 num_layers =
            (std::min)(copy.src_subresource.num_layers, copy.dst_subresource.num_layers);
        for (s32 layer = 0; layer < num_layers; ++layer) {
            @autoreleasepool {
                const NSRange src_levels = NSMakeRange(static_cast<NSUInteger>(src_level), 1);
                const NSRange src_slices =
                    NSMakeRange(static_cast<NSUInteger>(copy.src_subresource.base_layer + layer), 1);
                const NSUInteger dst_slice =
                    static_cast<NSUInteger>(copy.dst_subresource.base_layer + layer);
                MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                id<MTLTexture> source;
                id<MTLTexture> stencil_source;
                if (src_is_depth) {
                    id<MTLTexture> target = [dst.texture
                        newTextureViewWithPixelFormat:packed_format
                                          textureType:MTLTextureType2D
                                               levels:NSMakeRange(
                                                          static_cast<NSUInteger>(dst_level), 1)
                                               slices:NSMakeRange(dst_slice, 1)];
                    SetBlitAttachment(pass, target, 0, 0, covers_level);
                    source = [src.texture newTextureViewWithPixelFormat:src.Format()
                                                            textureType:MTLTextureType2D
                                                                 levels:src_levels
                                                                 slices:src_slices];
                    if (stencil) {
                        stencil_source = [src.texture newTextureViewWithPixelFormat:stencil_format
                                                                        textureType:MTLTextureType2D
                                                                             levels:src_levels
                                                                             slices:src_slices];
                    }
                } else {
                    SetBlitAttachment(pass, dst.texture, static_cast<NSUInteger>(dst_level),
                                      dst_slice, covers_level, stencil);
                    source = [src.texture newTextureViewWithPixelFormat:packed_format
                                                            textureType:MTLTextureType2D
                                                                 levels:src_levels
                                                                 slices:src_slices];
                }
                DrawBlitQuad(scheduler, pass, pipeline, depth_state, source, nil, params, rect,
                             stencil_source, layout_value);
            }
        }
    }
    return true;
}

void TextureCacheRuntime::CopyImageMSAA(Image& dst, Image& src,
                                        std::span<const ImageCopy> copies) {
    if (!dst.texture || !src.texture) {
        return;
    }
    if (!src.format_info.supported || !dst.format_info.supported) {
        WarnOnce(dst.info.format, "Unsupported MSAA image copy");
        return;
    }
    scheduler.FlushPendingClearsFor({dst.texture, src.texture},
                                    Scheduler::ClearFlushReason::ImageOperation);
    const bool msaa_source = src.texture.sampleCount > 1;
    const u32 num_samples = msaa_source ? src.info.num_samples : dst.info.num_samples;
    CopyMSAATextures(dst.texture, src.texture, num_samples, copies, dst.info.format);
}

void TextureCacheRuntime::CopyMSAATextures(id<MTLTexture> dst, id<MTLTexture> src,
                                           u32 num_samples, std::span<const ImageCopy> copies,
                                           PixelFormat guest_format) {
    // The images alias the same guest memory: every sample of the MSAA image is a texel of the
    // single sample one, so this is a sample (de)interleave and not a resolve.
    const bool msaa_source = src.sampleCount > 1;
    const bool msaa_target = dst.sampleCount > 1;
    const TexelType type = TexelTypeOf(dst.pixelFormat);
    const bool stencil = type == TexelType::Depth && HasStencilAspect(dst.pixelFormat) &&
                         HasStencilAspect(src.pixelFormat);
    const MTLPixelFormat stencil_format = stencil ? StencilViewFormat(src.pixelFormat)
                                                  : MTLPixelFormatInvalid;
    const auto shader = MsaaCopyShaderFor(type, msaa_source, stencil);
    if (msaa_source == msaa_target || !shader || type != TexelTypeOf(src.pixelFormat) ||
        (stencil && stencil_format == MTLPixelFormatInvalid) ||
        (dst.usage & MTLTextureUsageRenderTarget) == 0 ||
        src.textureType == MTLTextureType3D || dst.textureType == MTLTextureType3D) {
        WarnOnce(guest_format, "Unsupported MSAA image copy");
        return;
    }
    id<MTLRenderPipelineState> pipeline = BlitPipeline(*shader, dst.pixelFormat, dst.sampleCount);
    if (!pipeline) {
        return;
    }
    const auto [samples_x, samples_y] = VideoCommon::SamplesLog2(static_cast<int>(num_samples));
    const MTLTextureType source_type =
        msaa_source ? MTLTextureType2DMultisample : MTLTextureType2D;
    id<MTLDepthStencilState> depth_state = nil;
    if (type == TexelType::Depth) {
        depth_state = stencil ? blit_depth_stencil_write : blit_depth_write;
    }

    for (const ImageCopy& copy : copies) {
        const s32 src_level = copy.src_subresource.base_level;
        const s32 dst_level = copy.dst_subresource.base_level;
        const s32 level_width = static_cast<s32>(LevelExtent(dst.width, dst_level));
        const s32 level_height = static_cast<s32>(LevelExtent(dst.height, dst_level));
        const s32 x0 = std::clamp(copy.dst_offset.x, 0, level_width);
        const s32 y0 = std::clamp(copy.dst_offset.y, 0, level_height);
        const s32 x1 = (std::min)(x0 + static_cast<s32>(copy.extent.width), level_width);
        const s32 y1 = (std::min)(y0 + static_cast<s32>(copy.extent.height), level_height);
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }
        const MTLScissorRect rect{
            .x = static_cast<NSUInteger>(x0),
            .y = static_cast<NSUInteger>(y0),
            .width = static_cast<NSUInteger>(x1 - x0),
            .height = static_cast<NSUInteger>(y1 - y0),
        };
        const BlitParams params{
            .dst_offset = {copy.dst_offset.x, copy.dst_offset.y},
            .src_offset = {copy.src_offset.x, copy.src_offset.y},
            .scale = {1 << samples_x, 1 << samples_y},
            .src_size = {static_cast<s32>(LevelExtent(src.width, src_level)),
                         static_cast<s32>(LevelExtent(src.height, src_level))},
        };
        // Every texel and sample of the rectangle is written by the copy shaders.
        const bool covers_level = CoversLevel(rect, static_cast<NSUInteger>(level_width),
                                              static_cast<NSUInteger>(level_height));
        const s32 num_layers =
            (std::min)(copy.src_subresource.num_layers, copy.dst_subresource.num_layers);
        for (s32 layer = 0; layer < num_layers; ++layer) {
            @autoreleasepool {
                const NSUInteger src_layer =
                    static_cast<NSUInteger>(copy.src_subresource.base_layer + layer);
                const NSRange src_levels = NSMakeRange(static_cast<NSUInteger>(src_level), 1);
                const NSRange src_slices = NSMakeRange(src_layer, 1);
                id<MTLTexture> source = [src newTextureViewWithPixelFormat:src.pixelFormat
                                                               textureType:source_type
                                                                    levels:src_levels
                                                                    slices:src_slices];
                id<MTLTexture> stencil_source;
                if (stencil) {
                    stencil_source = [src newTextureViewWithPixelFormat:stencil_format
                                                            textureType:source_type
                                                                 levels:src_levels
                                                                 slices:src_slices];
                }
                MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                SetBlitAttachment(pass, dst, static_cast<NSUInteger>(dst_level),
                                  static_cast<NSUInteger>(copy.dst_subresource.base_layer + layer),
                                  covers_level, stencil);
                DrawBlitQuad(scheduler, pass, pipeline, depth_state, source, nil, params, rect,
                             stencil_source);
            }
        }
    }
}

id<MTLTexture> TextureCacheRuntime::ScratchTexture(MTLPixelFormat format, u32 width,
                                                   u32 height, u32 layers, u32 samples) {
    // Reusing a scratch texture is safe, Metal orders its tracked hazards across encoders.
    for (id<MTLTexture> scratch : scratch_textures) {
        if (scratch.pixelFormat == format && scratch.width == width && scratch.height == height &&
            scratch.arrayLength == layers && scratch.sampleCount == samples) {
            return scratch;
        }
    }
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    if (samples > 1) {
        descriptor.textureType =
            layers > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
    } else {
        descriptor.textureType = layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
    }
    descriptor.pixelFormat = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.arrayLength = layers;
    descriptor.sampleCount = samples;
    descriptor.storageMode = MTLStorageModePrivate;
    // Conversion passes render into scratch textures, uploads and blits read them.
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget |
                       MTLTextureUsagePixelFormatView;
    id<MTLTexture> scratch = [device.GetMTLDevice() newTextureWithDescriptor:descriptor];
    if (!scratch) {
        LOG_ERROR(Render_Metal, "Failed to allocate a {}x{} scratch texture", width, height);
        return nil;
    }
    // Keeps the most recently created ones, games use a few layouts per frame.
    static constexpr size_t MAX_SCRATCH_TEXTURES = 8;
    if (scratch_textures.size() >= MAX_SCRATCH_TEXTURES) {
        scratch_textures.erase(scratch_textures.begin());
    }
    scratch_textures.push_back(scratch);
    return scratch;
}

bool TextureCacheRuntime::UploadDepthStencilOnGpu(id<MTLTexture> target,
                                                  DepthStencilLayout layout, bool has_stencil,
                                                  u32 guest_bytes, id<MTLBuffer> buffer,
                                                  size_t offset, const BufferImageCopy& copy,
                                                  u32 width, u32 height, NSUInteger bytes_per_row,
                                                  NSUInteger layer_stride) {
    if ((target.usage & MTLTextureUsageRenderTarget) == 0 || target.sampleCount > 1 ||
        (guest_bytes != 4 && guest_bytes != 8)) {
        return false;
    }
    // The guest bytes are uploaded unchanged into a uint texture, a pass unpacks them.
    const MTLPixelFormat packed_format =
        guest_bytes == 8 ? MTLPixelFormatRG32Uint : MTLPixelFormatR32Uint;
    const BlitShader shader =
        has_stencil ? BlitShader::ColorToDepthStencil : BlitShader::ColorToDepth;
    id<MTLRenderPipelineState> pipeline = BlitPipeline(shader, target.pixelFormat, 1);
    const u32 num_layers = static_cast<u32>(copy.image_subresource.num_layers);
    id<MTLTexture> scratch =
        pipeline ? ScratchTexture(packed_format, width, height, num_layers) : nil;
    if (!scratch) {
        return false;
    }
    const NSUInteger bytes_per_image = IsLayered(scratch) ? bytes_per_row * height : 0;
    id<MTLBlitCommandEncoder> blit = scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    for (u32 layer = 0; layer < num_layers; ++layer) {
        [blit copyFromBuffer:buffer
                   sourceOffset:offset + copy.buffer_offset + layer * layer_stride
              sourceBytesPerRow:bytes_per_row
            sourceBytesPerImage:bytes_per_image
                     sourceSize:MTLSizeMake(width, height, 1)
                      toTexture:scratch
               destinationSlice:layer
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    }

    const s32 level = copy.image_subresource.base_level;
    const MTLScissorRect rect{
        .x = static_cast<NSUInteger>(copy.image_offset.x),
        .y = static_cast<NSUInteger>(copy.image_offset.y),
        .width = width,
        .height = height,
    };
    const bool covers_level =
        CoversLevel(rect, LevelExtent(target.width, level), LevelExtent(target.height, level));
    const BlitParams params{
        .dst_offset = {copy.image_offset.x, copy.image_offset.y},
        .src_size = {static_cast<s32>(width), static_cast<s32>(height)},
    };
    id<MTLDepthStencilState> depth_state =
        has_stencil ? blit_depth_stencil_write : blit_depth_write;
    for (u32 layer = 0; layer < num_layers; ++layer) {
        @autoreleasepool {
            id<MTLTexture> source = [scratch newTextureViewWithPixelFormat:packed_format
                                                               textureType:MTLTextureType2D
                                                                    levels:NSMakeRange(0, 1)
                                                                    slices:NSMakeRange(layer, 1)];
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            SetBlitAttachment(pass, target, static_cast<NSUInteger>(level),
                              static_cast<NSUInteger>(copy.image_subresource.base_layer) + layer,
                              covers_level, has_stencil);
            DrawBlitQuad(scheduler, pass, pipeline, depth_state, source, nil, params, rect, nil,
                         static_cast<u32>(layout));
        }
    }
    return true;
}

bool TextureCacheRuntime::DownloadDepthStencilOnGpu(id<MTLTexture> source,
                                                    DepthStencilLayout layout, bool has_stencil,
                                                    u32 guest_bytes, id<MTLBuffer> buffer,
                                                    size_t offset, const BufferImageCopy& copy,
                                                    u32 width, u32 height,
                                                    NSUInteger bytes_per_row,
                                                    NSUInteger layer_stride) {
    if (source.sampleCount > 1 || (guest_bytes != 4 && guest_bytes != 8)) {
        return false;
    }
    // A pass packs the texels into a uint texture in the guest layout, blitted to the buffer.
    const MTLPixelFormat packed_format =
        guest_bytes == 8 ? MTLPixelFormatRG32Uint : MTLPixelFormatR32Uint;
    const MTLPixelFormat stencil_format = StencilViewFormat(source.pixelFormat);
    if (has_stencil && stencil_format == MTLPixelFormatInvalid) {
        return false;
    }
    const BlitShader shader =
        has_stencil ? BlitShader::DepthStencilToColor : BlitShader::DepthToColor;
    id<MTLRenderPipelineState> pipeline = BlitPipeline(shader, packed_format, 1);
    const u32 num_layers = static_cast<u32>(copy.image_subresource.num_layers);
    id<MTLTexture> scratch =
        pipeline ? ScratchTexture(packed_format, width, height, num_layers) : nil;
    if (!scratch) {
        return false;
    }
    const s32 level = copy.image_subresource.base_level;
    const MTLScissorRect rect{.x = 0, .y = 0, .width = width, .height = height};
    const BlitParams params{
        .src_offset = {copy.image_offset.x, copy.image_offset.y},
        .src_size = {static_cast<s32>(LevelExtent(source.width, level)),
                     static_cast<s32>(LevelExtent(source.height, level))},
    };
    for (u32 layer = 0; layer < num_layers; ++layer) {
        @autoreleasepool {
            const NSRange levels = NSMakeRange(static_cast<NSUInteger>(level), 1);
            const NSRange slices =
                NSMakeRange(static_cast<NSUInteger>(copy.image_subresource.base_layer) + layer, 1);
            id<MTLTexture> depth = [source newTextureViewWithPixelFormat:source.pixelFormat
                                                             textureType:MTLTextureType2D
                                                                  levels:levels
                                                                  slices:slices];
            id<MTLTexture> stencil;
            if (has_stencil) {
                stencil = [source newTextureViewWithPixelFormat:stencil_format
                                                    textureType:MTLTextureType2D
                                                         levels:levels
                                                          slices:slices];
            }
            id<MTLTexture> target = [scratch newTextureViewWithPixelFormat:packed_format
                                                               textureType:MTLTextureType2D
                                                                    levels:NSMakeRange(0, 1)
                                                                    slices:NSMakeRange(layer, 1)];
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            SetBlitAttachment(pass, target, 0, 0, true);
            DrawBlitQuad(scheduler, pass, pipeline, nil, depth, nil, params, rect, stencil,
                         static_cast<u32>(layout));
        }
    }
    const NSUInteger bytes_per_image = IsLayered(scratch) ? bytes_per_row * height : 0;
    id<MTLBlitCommandEncoder> blit = scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    for (u32 layer = 0; layer < num_layers; ++layer) {
        [blit copyFromTexture:scratch
                         sourceSlice:layer
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(width, height, 1)
                            toBuffer:buffer
                   destinationOffset:offset + copy.buffer_offset + layer * layer_stride
              destinationBytesPerRow:bytes_per_row
            destinationBytesPerImage:bytes_per_image];
    }
    return true;
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) {
    if (dst.Format() == src.Format()) {
        return false;
    }
    const auto is_depth_stencil = [](const Image& image) {
        return GetFormatType(image.info.format) != SurfaceType::ColorTexture;
    };
    return is_depth_stencil(dst) || is_depth_stencil(src);
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const ImageCopy> copies) {
    CopyThroughBuffer(dst, src, copies);
}

void TextureCacheRuntime::ConvertImage(Framebuffer*, ImageView& dst_view, ImageView& src_view) {
    // TODO: Needs the render pipelines of the next milestone.
    WarnOnce(src_view.format, "Image view conversions are not implemented");
    (void)dst_view;
}

void TextureCacheRuntime::BlitImage(Framebuffer*, ImageView& dst, ImageView& src,
                                    const Region2D& dst_region, const Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation operation) {
    using Tegra::Engines::Fermi2D;
    id<MTLTexture> src_texture = src.ImageHandle();
    id<MTLTexture> dst_texture = dst.ImageHandle();
    if (!src_texture || !dst_texture) {
        return;
    }
    scheduler.FlushPendingClearsFor({dst_texture, src_texture},
                                    Scheduler::ClearFlushReason::ImageOperation);
    const s32 src_width = src_region.end.x - src_region.start.x;
    const s32 src_height = src_region.end.y - src_region.start.y;
    const s32 dst_width = dst_region.end.x - dst_region.start.x;
    const s32 dst_height = dst_region.end.y - dst_region.start.y;
    const bool is_copy = operation == Fermi2D::Operation::SrcCopy ||
                         operation == Fermi2D::Operation::SrcCopyAnd;
    if (!is_copy || src_width != dst_width || src_height != dst_height || src_width <= 0 ||
        src_height <= 0 || src_texture.pixelFormat != dst_texture.pixelFormat ||
        src_texture.sampleCount != dst_texture.sampleCount) {
        if (!is_copy) {
            WarnOnce(dst.format, "Blit operations other than copies are drawn as copies");
        }
        BlitWithShader(dst, src, dst_region, src_region, filter);
        return;
    }
    const s32 src_level = src.range.base.level;
    const s32 dst_level = dst.range.base.level;
    const bool src_3d = src_texture.textureType == MTLTextureType3D;
    const bool dst_3d = dst_texture.textureType == MTLTextureType3D;
    const u32 width = (std::min)({static_cast<u32>(src_width),
                                  LevelExtent(src_texture.width, src_level) -
                                      static_cast<u32>(src_region.start.x),
                                  LevelExtent(dst_texture.width, dst_level) -
                                      static_cast<u32>(dst_region.start.x)});
    const u32 height = (std::min)({static_cast<u32>(src_height),
                                   LevelExtent(src_texture.height, src_level) -
                                       static_cast<u32>(src_region.start.y),
                                   LevelExtent(dst_texture.height, dst_level) -
                                       static_cast<u32>(dst_region.start.y)});
    scheduler.EndEncoding(Scheduler::PassEnd::ImageBlit);
    scheduler.CountSurfaceCopy();
    [scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit) copyFromTexture:src_texture
                                 sourceSlice:src_3d ? 0 : src.range.base.layer
                                 sourceLevel:src_level
                                sourceOrigin:MTLOriginMake(src_region.start.x, src_region.start.y,
                                                           src_3d ? src.range.base.layer : 0)
                                  sourceSize:MTLSizeMake(width, height, 1)
                                   toTexture:dst_texture
                            destinationSlice:dst_3d ? 0 : dst.range.base.layer
                            destinationLevel:dst_level
                           destinationOrigin:MTLOriginMake(dst_region.start.x, dst_region.start.y,
                                                           dst_3d ? dst.range.base.layer : 0)];
}

void TextureCacheRuntime::BlitWithShader(ImageView& dst, ImageView& src,
                                         const Region2D& dst_region, const Region2D& src_region,
                                         Tegra::Engines::Fermi2D::Filter filter) {
    id<MTLTexture> target = dst.RenderTarget();
    id<MTLTexture> source = src.Handle(Shader::TextureType::Color2D);
    if (!target || !source) {
        WarnOnce(dst.format, "Blit image is not renderable or not sampleable");
        return;
    }
    const TexelType type = TexelTypeOf(target.pixelFormat);
    const auto shader = BlitShaderFor(type, source.sampleCount > 1);
    if (!shader || type != TexelTypeOf(source.pixelFormat)) {
        WarnOnce(dst.format, "Blits between these formats are not implemented");
        return;
    }

    // Flips are moved to the source rectangle, the quad always covers a positive area.
    s32 dx0 = dst_region.start.x;
    s32 dx1 = dst_region.end.x;
    s32 dy0 = dst_region.start.y;
    s32 dy1 = dst_region.end.y;
    f32 sx0 = static_cast<f32>(src_region.start.x);
    f32 sx1 = static_cast<f32>(src_region.end.x);
    f32 sy0 = static_cast<f32>(src_region.start.y);
    f32 sy1 = static_cast<f32>(src_region.end.y);
    if (dx0 > dx1) {
        std::swap(dx0, dx1);
        std::swap(sx0, sx1);
    }
    if (dy0 > dy1) {
        std::swap(dy0, dy1);
        std::swap(sy0, sy1);
    }
    if (!ClipBlitSpan(dx0, dx1, sx0, sx1, static_cast<s32>(target.width)) ||
        !ClipBlitSpan(dy0, dy1, sy0, sy1, static_cast<s32>(target.height))) {
        return;
    }
    id<MTLRenderPipelineState> pipeline =
        BlitPipeline(*shader, target.pixelFormat, target.sampleCount);
    if (!pipeline) {
        return;
    }

    @autoreleasepool {
        if (src.ImageHandle() == dst.ImageHandle()) {
            // A pass can't sample its own attachment, read from a copy of the source level.
            id<MTLTexture> copy = ScratchTexture(
                source.pixelFormat, static_cast<u32>(source.width),
                static_cast<u32>(source.height), 1, static_cast<u32>(source.sampleCount));
            if (!copy) {
                return;
            }
            scheduler.EndEncoding(Scheduler::PassEnd::ImageBlit);
            [scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit) copyFromTexture:source
                                         sourceSlice:0
                                         sourceLevel:0
                                        sourceOrigin:MTLOriginMake(0, 0, 0)
                                          sourceSize:MTLSizeMake(source.width, source.height, 1)
                                           toTexture:copy
                                    destinationSlice:0
                                    destinationLevel:0
                                   destinationOrigin:MTLOriginMake(0, 0, 0)];
            source = copy;
        }

        const NSUInteger slice =
            target.textureType == MTLTextureType3D
                ? (std::min<NSUInteger>)(static_cast<NSUInteger>(dst.range.base.layer),
                                         target.depth - 1)
                : 0;
        const BlitParams params{
            .src_rect = {sx0, sy0, sx1, sy1},
            .src_size = {static_cast<s32>(source.width), static_cast<s32>(source.height)},
        };
        const MTLScissorRect rect{
            .x = static_cast<NSUInteger>(dx0),
            .y = static_cast<NSUInteger>(dy0),
            .width = static_cast<NSUInteger>(dx1 - dx0),
            .height = static_cast<NSUInteger>(dy1 - dy0),
        };
        // Blits covering the whole target don't need its old contents.
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        SetBlitAttachment(pass, target, 0, slice, CoversLevel(rect, target.width, target.height));
        id<MTLSamplerState> sampler = nil;
        if (*shader == BlitShader::SampleFloat) {
            sampler = filter == Tegra::Engines::Fermi2D::Filter::Bilinear ? blit_linear_sampler
                                                                          : blit_nearest_sampler;
        }
        id<MTLDepthStencilState> depth_state = type == TexelType::Depth ? blit_depth_write : nil;
        DrawBlitQuad(scheduler, pass, pipeline, depth_state, source, sampler, params, rect);
    }
}

Image::Image(TextureCacheRuntime& runtime_, const ImageInfo& info_, GPUVAddr gpu_addr_,
             VAddr cpu_addr_)
    : VideoCommon::ImageBase(info_, gpu_addr_, cpu_addr_), runtime{&runtime_} {
    runtime->MarkImagePages(*this);
    const Device& device = runtime->device;
    format_info = SurfaceFormat(device, info.format);
    depth_layout = DepthLayoutOf(info.format);
    has_depth = HasDepthAspect(format_info.format);
    has_stencil = HasStencilAspect(format_info.format);
    if (NeedsConversion(device, info.format)) {
        flags |= ImageFlagBits::Converted;
        flags |= ImageFlagBits::CostlyLoad;
        const HostBlock block = ConvertedBlock(format_info.format);
        staging_layout = BlockLayout{block.width, block.height, block.bytes};
    } else {
        staging_layout = BlockLayout{
            DefaultBlockWidth(info.format),
            DefaultBlockHeight(info.format),
            BytesPerBlock(info.format),
        };
    }
    if (!format_info.supported) {
        WarnOnce(info.format, "Pixel format has no Metal equivalent, contents are not uploaded");
    }
    if (info.type == ImageType::Buffer) {
        return;
    }

    const auto [samples_x, samples_y] =
        VideoCommon::SamplesLog2(static_cast<int>(info.num_samples));
    NSUInteger sample_count = info.num_samples;
    while (sample_count > 1 && ![device.GetMTLDevice() supportsTextureSampleCount:sample_count]) {
        sample_count >>= 1;
    }
    if (sample_count != info.num_samples) {
        LOG_WARNING(Render_Metal, "{}x MSAA is not supported, using {}x", info.num_samples,
                    sample_count);
    }
    const u32 layers = static_cast<u32>((std::max)(info.resources.layers, 1));
    const bool is_3d = info.type == ImageType::e3D;
    const bool is_msaa = sample_count > 1;

    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    // 1D images are stored as 2D textures: Metal 1D textures can't have mipmaps.
    if (is_3d) {
        descriptor.textureType = MTLTextureType3D;
    } else if (is_msaa) {
        descriptor.textureType =
            layers > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
    } else {
        descriptor.textureType = layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
    }
    descriptor.pixelFormat = format_info.format;
    descriptor.width = (std::max)(info.size.width >> samples_x, 1U);
    descriptor.height = (std::max)(info.size.height >> samples_y, 1U);
    descriptor.depth = is_3d ? (std::max)(info.size.depth, 1U) : 1;
    descriptor.mipmapLevelCount =
        is_msaa ? 1 : static_cast<NSUInteger>((std::max)(info.resources.levels, 1));
    descriptor.arrayLength = is_3d ? 1 : layers;
    descriptor.sampleCount = sample_count;
    descriptor.storageMode = MTLStorageModePrivate;
    // PixelFormatView allows sRGB, stencil and same size class views of the image.
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (format_info.attachable) {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (format_info.storage && !is_msaa) {
        usage |= MTLTextureUsageShaderWrite;
    }
    descriptor.usage = usage;

    texture = [device.GetMTLDevice() newTextureWithDescriptor:descriptor];
    if (!texture) {
        LOG_ERROR(Render_Metal, "Failed to create a {}x{}x{} texture with {} layers, format {}",
                  descriptor.width, descriptor.height, descriptor.depth, layers, info.format);
    }
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

Image::~Image() = default;

id<MTLTexture> Image::ReinterpretView(MTLPixelFormat format) {
    if (!texture) {
        return nil;
    }
    if (format == format_info.format) {
        return texture;
    }
    if (reinterpret_view && reinterpret_format == format) {
        return reinterpret_view;
    }
    @autoreleasepool {
        reinterpret_view = [texture newTextureViewWithPixelFormat:format];
    }
    reinterpret_format = reinterpret_view ? format : MTLPixelFormatInvalid;
    return reinterpret_view;
}

void Image::UploadMemory(id<MTLBuffer> buffer, size_t offset,
                         std::span<const BufferImageCopy> copies) {
    if (!texture || !buffer || copies.empty()) {
        return;
    }
    if (!format_info.supported) {
        WarnOnce(info.format, "Skipping upload of an unsupported image");
        return;
    }
    runtime->scheduler.CountImageUpload();
    runtime->scheduler.FlushPendingClearsFor({texture},
                                             Scheduler::ClearFlushReason::ImageOperation);
    if (texture.sampleCount > 1) {
        UploadMSAA(buffer, offset, copies);
        return;
    }
    // Ordered after earlier blits touching this image; uploads of other images share the
    // encoder.
    (void)runtime->scheduler.BlitEncoderFor(blit_generation);
    for (const BufferImageCopy& copy : copies) {
        if (depth_layout != DepthStencilLayout::Native) {
            UploadDepthStencil(texture, buffer, offset, copy);
        } else {
            EncodeUpload(texture, buffer, offset, copy);
        }
    }
}

void Image::UploadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    UploadMemory(map.buffer, map.offset, copies);
}

void Image::UploadMSAA(id<MTLBuffer> buffer, size_t offset,
                       std::span<const BufferImageCopy> copies) {
    // Guest MSAA images store each sample as a texel, the scratch texture has the guest size.
    id<MTLTexture> scratch = runtime->ScratchTexture(
        format_info.format, (std::max)(info.size.width, 1U), (std::max)(info.size.height, 1U),
        static_cast<u32>((std::max)(info.resources.layers, 1)));
    if (!scratch) {
        return;
    }
    const auto [samples_x, samples_y] =
        VideoCommon::SamplesLog2(static_cast<int>(info.num_samples));
    std::vector<ImageCopy> image_copies;
    image_copies.reserve(copies.size());

    (void)runtime->scheduler.BlitEncoderFor(blit_generation);
    for (const BufferImageCopy& copy : copies) {
        if (depth_layout != DepthStencilLayout::Native) {
            UploadDepthStencil(scratch, buffer, offset, copy);
        } else {
            EncodeUpload(scratch, buffer, offset, copy);
        }
        image_copies.push_back(ImageCopy{
            .src_subresource = copy.image_subresource,
            .dst_subresource = copy.image_subresource,
            .src_offset = {.x = copy.image_offset.x, .y = copy.image_offset.y, .z = 0},
            .dst_offset = {.x = copy.image_offset.x >> samples_x,
                           .y = copy.image_offset.y >> samples_y,
                           .z = 0},
            .extent = {.width = (std::max)(copy.image_extent.width >> samples_x, 1U),
                       .height = (std::max)(copy.image_extent.height >> samples_y, 1U),
                       .depth = 1},
        });
    }
    runtime->CopyMSAATextures(texture, scratch, info.num_samples, image_copies, info.format);
}

void Image::DownloadMemory(id<MTLBuffer> buffer, size_t offset,
                           std::span<const BufferImageCopy> copies) {
    if (!texture || !buffer || copies.empty()) {
        return;
    }
    if (!format_info.supported || texture.sampleCount > 1) {
        WarnOnce(info.format, "Skipping download of an unsupported image");
        return;
    }
    runtime->scheduler.CountImageDownload();
    runtime->scheduler.FlushPendingClearsFor({texture},
                                             Scheduler::ClearFlushReason::ImageOperation);
    // Ordered after earlier blits touching this image; downloads of other images share the
    // encoder.
    (void)runtime->scheduler.BlitEncoderFor(blit_generation);
    for (const BufferImageCopy& copy : copies) {
        if (depth_layout != DepthStencilLayout::Native) {
            DownloadDepthStencil(buffer, offset, copy);
        } else {
            EncodeDownload(buffer, offset, copy);
        }
    }
    runtime->download_tick = runtime->scheduler.CurrentTick();
}

void Image::DownloadMemory(std::span<id<MTLBuffer>> buffers, std::span<size_t> offsets,
                           std::span<const BufferImageCopy> copies) {
    for (size_t i = 0; i < buffers.size() && i < offsets.size(); ++i) {
        DownloadMemory(buffers[i], offsets[i], copies);
    }
}

void Image::DownloadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    DownloadMemory(map.buffer, map.offset, copies);
}

namespace {

struct CopyLayout {
    u32 width;
    u32 height;
    u32 depth;
    NSUInteger bytes_per_row;
    NSUInteger bytes_per_image;
    NSUInteger layer_stride;
};

template <typename Layout>
CopyLayout MakeCopyLayout(id<MTLTexture> texture, const Layout& block,
                          const BufferImageCopy& copy) {
    const s32 level = copy.image_subresource.base_level;
    const u32 row_length = copy.buffer_row_length != 0 ? copy.buffer_row_length
                                                       : copy.image_extent.width;
    const u32 image_height = copy.buffer_image_height != 0 ? copy.buffer_image_height
                                                           : copy.image_extent.height;
    const NSUInteger bytes_per_row =
        static_cast<NSUInteger>(Common::DivCeil(row_length, block.width)) * block.bytes;
    const NSUInteger bytes_per_image =
        bytes_per_row * Common::DivCeil(image_height, block.height);
    // Metal rejects regions outside of the mip level; compressed regions may end at its edge.
    const u32 width = (std::min)(copy.image_extent.width,
                                 LevelExtent(texture.width, level) -
                                     static_cast<u32>(copy.image_offset.x));
    const u32 height = (std::min)(copy.image_extent.height,
                                  LevelExtent(texture.height, level) -
                                      static_cast<u32>(copy.image_offset.y));
    const u32 depth = texture.textureType == MTLTextureType3D
                          ? (std::min)(copy.image_extent.depth,
                                       LevelExtent(texture.depth, level) -
                                           static_cast<u32>(copy.image_offset.z))
                          : 1;
    return CopyLayout{
        .width = width,
        .height = height,
        .depth = depth,
        .bytes_per_row = bytes_per_row,
        .bytes_per_image = bytes_per_image,
        .layer_stride = bytes_per_image * (std::max)(copy.image_extent.depth, 1U),
    };
}

} // Anonymous namespace

void Image::EncodeUpload(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                         const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(target, staging_layout, copy);
    if (layout.width == 0 || layout.height == 0 || layout.depth == 0) {
        return;
    }
    const bool is_3d = target.textureType == MTLTextureType3D;
    const NSUInteger bytes_per_image = is_3d || IsLayered(target) ? layout.bytes_per_image : 0;
    id<MTLBlitCommandEncoder> blit = runtime->scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    for (s32 layer = 0; layer < copy.image_subresource.num_layers; ++layer) {
        [blit copyFromBuffer:buffer
                   sourceOffset:offset + copy.buffer_offset + layer * layout.layer_stride
              sourceBytesPerRow:layout.bytes_per_row
            sourceBytesPerImage:bytes_per_image
                     sourceSize:MTLSizeMake(layout.width, layout.height, layout.depth)
                      toTexture:target
               destinationSlice:is_3d ? 0 : copy.image_subresource.base_layer + layer
               destinationLevel:copy.image_subresource.base_level
              destinationOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                              is_3d ? copy.image_offset.z : 0)];
    }
}

void Image::EncodeDownload(id<MTLBuffer> buffer, size_t offset, const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(texture, staging_layout, copy);
    if (layout.width == 0 || layout.height == 0 || layout.depth == 0) {
        return;
    }
    const bool is_3d = texture.textureType == MTLTextureType3D;
    const NSUInteger bytes_per_image = is_3d || IsLayered(texture) ? layout.bytes_per_image : 0;
    id<MTLBlitCommandEncoder> blit = runtime->scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    for (s32 layer = 0; layer < copy.image_subresource.num_layers; ++layer) {
        [blit copyFromTexture:texture
                         sourceSlice:is_3d ? 0 : copy.image_subresource.base_layer + layer
                         sourceLevel:copy.image_subresource.base_level
                        sourceOrigin:MTLOriginMake(copy.image_offset.x, copy.image_offset.y,
                                                   is_3d ? copy.image_offset.z : 0)
                          sourceSize:MTLSizeMake(layout.width, layout.height, layout.depth)
                            toBuffer:buffer
                   destinationOffset:offset + copy.buffer_offset + layer * layout.layer_stride
              destinationBytesPerRow:layout.bytes_per_row
            destinationBytesPerImage:bytes_per_image];
    }
}

void Image::UploadDepthStencil(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                               const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(target, staging_layout, copy);
    if (layout.width == 0 || layout.height == 0 || copy.image_subresource.num_layers <= 0) {
        return;
    }
    if (runtime->UploadDepthStencilOnGpu(target, depth_layout, has_stencil, staging_layout.bytes,
                                         buffer, offset, copy, layout.width, layout.height,
                                         layout.bytes_per_row, layout.layer_stride)) {
        return;
    }
    UploadDepthStencilCpu(target, buffer, offset, copy);
}

void Image::UploadDepthStencilCpu(id<MTLTexture> target, id<MTLBuffer> buffer, size_t offset,
                                  const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(target, staging_layout, copy);
    const u32 num_layers = static_cast<u32>(copy.image_subresource.num_layers);
    if (layout.width == 0 || layout.height == 0 || num_layers == 0) {
        return;
    }
    const size_t texels = static_cast<size_t>(layout.width) * layout.height;
    const size_t depth_plane = texels * sizeof(f32);
    const size_t stencil_plane = has_stencil ? texels : 0;
    const size_t layer_size = depth_plane + stencil_plane;
    const StagingBufferRef staging =
        runtime->staging_pool.Request(layer_size * num_layers, MemoryUsage::Upload);

    // The guest bytes were written by the CPU into shared memory, convert them in place.
    const u8* const guest = static_cast<const u8*>(buffer.contents) + offset + copy.buffer_offset;
    for (u32 layer = 0; layer < num_layers; ++layer) {
        u8* const plane = staging.mapped_span.data() + layer * layer_size;
        for (u32 y = 0; y < layout.height; ++y) {
            const u8* row = guest + layer * layout.layer_stride + y * layout.bytes_per_row;
            for (u32 x = 0; x < layout.width; ++x) {
                const size_t texel = static_cast<size_t>(y) * layout.width + x;
                f32 depth = 0.0f;
                u8 stencil = 0;
                u32 packed;
                std::memcpy(&packed, row + x * staging_layout.bytes, sizeof(packed));
                switch (depth_layout) {
                case DepthStencilLayout::D24S8:
                    depth = static_cast<f32>((packed >> 8) / UNORM24_MAX);
                    stencil = static_cast<u8>(packed);
                    break;
                case DepthStencilLayout::S8D24:
                    depth = static_cast<f32>((packed & 0xFFFFFF) / UNORM24_MAX);
                    stencil = static_cast<u8>(packed >> 24);
                    break;
                case DepthStencilLayout::X8D24:
                    depth = static_cast<f32>((packed & 0xFFFFFF) / UNORM24_MAX);
                    break;
                case DepthStencilLayout::D32S8:
                    std::memcpy(&depth, &packed, sizeof(depth));
                    stencil = row[x * staging_layout.bytes + 4];
                    break;
                case DepthStencilLayout::Native:
                    break;
                }
                std::memcpy(plane + texel * sizeof(f32), &depth, sizeof(depth));
                if (has_stencil) {
                    plane[depth_plane + texel] = stencil;
                }
            }
        }
    }

    id<MTLBlitCommandEncoder> blit = runtime->scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    const bool combined = has_depth && has_stencil;
    for (u32 layer = 0; layer < num_layers; ++layer) {
        const NSUInteger base = staging.offset + layer * layer_size;
        const NSUInteger slice = copy.image_subresource.base_layer + layer;
        const MTLOrigin origin = MTLOriginMake(copy.image_offset.x, copy.image_offset.y, 0);
        const MTLSize size = MTLSizeMake(layout.width, layout.height, 1);
        [blit copyFromBuffer:staging.buffer
                   sourceOffset:base
              sourceBytesPerRow:layout.width * sizeof(f32)
            sourceBytesPerImage:IsLayered(target) ? depth_plane : 0
                     sourceSize:size
                      toTexture:target
               destinationSlice:slice
               destinationLevel:copy.image_subresource.base_level
              destinationOrigin:origin
                        options:combined ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
        if (has_stencil) {
            [blit copyFromBuffer:staging.buffer
                       sourceOffset:base + depth_plane
                  sourceBytesPerRow:layout.width
                sourceBytesPerImage:IsLayered(target) ? texels : 0
                         sourceSize:size
                          toTexture:target
                   destinationSlice:slice
                   destinationLevel:copy.image_subresource.base_level
                  destinationOrigin:origin
                            options:MTLBlitOptionStencilFromDepthStencil];
        }
    }
}

void Image::DownloadDepthStencil(id<MTLBuffer> buffer, size_t offset, const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(texture, staging_layout, copy);
    if (layout.width == 0 || layout.height == 0 || copy.image_subresource.num_layers <= 0) {
        return;
    }
    if (runtime->DownloadDepthStencilOnGpu(texture, depth_layout, has_stencil,
                                           staging_layout.bytes, buffer, offset, copy,
                                           layout.width, layout.height, layout.bytes_per_row,
                                           layout.layer_stride)) {
        return;
    }
    DownloadDepthStencilCpu(buffer, offset, copy);
}

void Image::DownloadDepthStencilCpu(id<MTLBuffer> buffer, size_t offset,
                                    const BufferImageCopy& copy) {
    const CopyLayout layout = MakeCopyLayout(texture, staging_layout, copy);
    const u32 num_layers = static_cast<u32>(copy.image_subresource.num_layers);
    if (layout.width == 0 || layout.height == 0 || num_layers == 0) {
        return;
    }
    const size_t texels = static_cast<size_t>(layout.width) * layout.height;
    const size_t depth_plane = texels * sizeof(f32);
    const size_t stencil_plane = has_stencil ? texels : 0;
    const size_t layer_size = depth_plane + stencil_plane;
    const StagingBufferRef staging =
        runtime->staging_pool.Request(layer_size * num_layers, MemoryUsage::Download);

    id<MTLBlitCommandEncoder> blit = runtime->scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    const bool combined = has_depth && has_stencil;
    for (u32 layer = 0; layer < num_layers; ++layer) {
        const NSUInteger base = staging.offset + layer * layer_size;
        const NSUInteger slice = copy.image_subresource.base_layer + layer;
        const MTLOrigin origin = MTLOriginMake(copy.image_offset.x, copy.image_offset.y, 0);
        const MTLSize size = MTLSizeMake(layout.width, layout.height, 1);
        [blit copyFromTexture:texture
                         sourceSlice:slice
                         sourceLevel:copy.image_subresource.base_level
                        sourceOrigin:origin
                          sourceSize:size
                            toBuffer:staging.buffer
                   destinationOffset:base
              destinationBytesPerRow:layout.width * sizeof(f32)
            destinationBytesPerImage:IsLayered(texture) ? depth_plane : 0
                             options:combined ? MTLBlitOptionDepthFromDepthStencil
                                              : MTLBlitOptionNone];
        if (has_stencil) {
            [blit copyFromTexture:texture
                             sourceSlice:slice
                             sourceLevel:copy.image_subresource.base_level
                            sourceOrigin:origin
                              sourceSize:size
                                toBuffer:staging.buffer
                       destinationOffset:base + depth_plane
                  destinationBytesPerRow:layout.width
                destinationBytesPerImage:IsLayered(texture) ? texels : 0
                                 options:MTLBlitOptionStencilFromDepthStencil];
        }
    }

    // Packs the planes back into the guest layout once the copies completed, before the tick
    // is reported as done so that waiters always observe the converted bytes.
    id<MTLBuffer> planes = staging.buffer;
    const size_t planes_offset = staging.offset;
    const size_t guest_offset = offset + copy.buffer_offset;
    const DepthStencilLayout packing = depth_layout;
    const u32 guest_bytes = staging_layout.bytes;
    const bool stencil = has_stencil;
    runtime->scheduler.OnCompletion([=] {
        u8* const guest = static_cast<u8*>(buffer.contents) + guest_offset;
        const u8* const source = static_cast<const u8*>(planes.contents) + planes_offset;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            const u8* const plane = source + layer * layer_size;
            for (u32 y = 0; y < layout.height; ++y) {
                u8* const row = guest + layer * layout.layer_stride + y * layout.bytes_per_row;
                for (u32 x = 0; x < layout.width; ++x) {
                    const size_t texel = static_cast<size_t>(y) * layout.width + x;
                    f32 depth;
                    std::memcpy(&depth, plane + texel * sizeof(f32), sizeof(depth));
                    const u32 stencil_value = stencil ? plane[depth_plane + texel] : 0;
                    u8* const out = row + x * guest_bytes;
                    u32 packed = 0;
                    switch (packing) {
                    case DepthStencilLayout::D24S8:
                        packed = (ToUnorm24(depth) << 8) | stencil_value;
                        break;
                    case DepthStencilLayout::S8D24:
                        packed = ToUnorm24(depth) | (stencil_value << 24);
                        break;
                    case DepthStencilLayout::X8D24:
                        packed = ToUnorm24(depth);
                        break;
                    case DepthStencilLayout::D32S8: {
                        std::memcpy(&packed, &depth, sizeof(packed));
                        const u32 high = stencil_value;
                        std::memcpy(out + 4, &high, sizeof(high));
                        break;
                    }
                    case DepthStencilLayout::Native:
                        break;
                    }
                    std::memcpy(out, &packed, sizeof(packed));
                }
            }
        }
    });
}

ImageView::ImageView(TextureCacheRuntime& runtime, const ImageViewInfo& info, ImageId image_id_,
                     Image& image, const SlotVector<Image>&)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr},
      image_handle{image.Handle()} {
    if (!image_handle) {
        return;
    }
    samples = static_cast<u32>(image_handle.sampleCount);
    const Device& device = runtime.device;
    const MTLPixelFormat image_format = image.Format();
    const FormatInfo view_info = SurfaceFormat(device, format);
    MTLPixelFormat pixel_format = view_info.format;
    if (!view_info.supported || !image.GetFormatInfo().supported ||
        !IsViewCompatible(image.info.format, image_format, format, pixel_format)) {
        if (image_format != pixel_format) {
            WarnOnce(format, "Incompatible view format, using the image format");
        }
        pixel_format = image_format;
    }

    const bool is_depth_stencil = HasDepthAspect(image_format) || HasStencilAspect(image_format);
    bool reads_stencil = false;
    if (is_depth_stencil && !info.IsRenderTarget()) {
        const auto swizzle = info.Swizzle();
        const bool any_r =
            std::ranges::any_of(swizzle, [](SwizzleSource s) { return s == SwizzleSource::R; });
        switch (format) {
        case PixelFormat::D24_UNORM_S8_UINT:
        case PixelFormat::D32_FLOAT_S8_UINT:
            // R = depth, G = stencil
            reads_stencil = !any_r;
            break;
        case PixelFormat::S8_UINT_D24_UNORM:
            // R = stencil, G = depth
            reads_stencil = any_r;
            break;
        case PixelFormat::S8_UINT:
            reads_stencil = true;
            break;
        default:
            break;
        }
    }
    const MTLPixelFormat sample_format =
        reads_stencil && image_format == MTLPixelFormatDepth32Float_Stencil8
            ? MTLPixelFormatX32_Stencil8
            : pixel_format;
    view_format = pixel_format;
    samples_stencil = reads_stencil;

    std::array<SwizzleSource, 4> swizzle{SwizzleSource::R, SwizzleSource::G, SwizzleSource::B,
                                         SwizzleSource::A};
    if (!info.IsRenderTarget() && !is_depth_stencil) {
        swizzle = info.Swizzle();
        TransformSwizzle(format, swizzle);
    }
    const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        ToMetalSwizzle(swizzle[0]), ToMetalSwizzle(swizzle[1]), ToMetalSwizzle(swizzle[2]),
        ToMetalSwizzle(swizzle[3]));
    const bool identity = swizzle[0] == SwizzleSource::R && swizzle[1] == SwizzleSource::G &&
                          swizzle[2] == SwizzleSource::B && swizzle[3] == SwizzleSource::A;

    const MTLTextureType parent_type = image_handle.textureType;
    const NSUInteger parent_levels = image_handle.mipmapLevelCount;
    const NSUInteger parent_layers =
        parent_type == MTLTextureType3D ? 1 : image_handle.arrayLength;
    const NSUInteger base_level = (std::min<NSUInteger>)(range.base.level, parent_levels - 1);
    const NSUInteger num_levels =
        (std::clamp<NSUInteger>)(range.extent.levels, 1, parent_levels - base_level);
    const NSUInteger base_layer = (std::min<NSUInteger>)(range.base.layer, parent_layers - 1);
    const NSUInteger num_layers =
        (std::clamp<NSUInteger>)(range.extent.layers, 1, parent_layers - base_layer);

    const auto make = [&](MTLTextureType type, NSUInteger layers, MTLPixelFormat view_pixel_format,
                          NSUInteger levels, bool swizzled) -> id<MTLTexture> {
        const NSRange level_range = NSMakeRange(base_level, levels);
        const NSRange slice_range = NSMakeRange(base_layer, layers);
        if (swizzled && !identity) {
            return [image_handle newTextureViewWithPixelFormat:view_pixel_format
                                                   textureType:type
                                                        levels:level_range
                                                        slices:slice_range
                                                       swizzle:channels];
        }
        return [image_handle newTextureViewWithPixelFormat:view_pixel_format
                                               textureType:type
                                                    levels:level_range
                                                    slices:slice_range];
    };
    const auto set = [&](Shader::TextureType type, id<MTLTexture> view) {
        views[static_cast<size_t>(type)] = view;
    };

    using Shader::TextureType;
    switch (parent_type) {
    case MTLTextureType3D:
        if (info.type == ImageViewType::e3D) {
            set(TextureType::Color3D, make(MTLTextureType3D, 1, sample_format, num_levels, true));
        } else {
            LOG_WARNING(Render_Metal, "2D views of 3D images are not supported by Metal");
        }
        break;
    case MTLTextureType2DMultisample:
    case MTLTextureType2DMultisampleArray: {
        id<MTLTexture> single = make(MTLTextureType2DMultisample, 1, sample_format, 1, true);
        set(TextureType::Color2D, single);
        set(TextureType::Color1D, single);
        set(TextureType::Color2DRect, single);
        if (parent_type == MTLTextureType2DMultisampleArray) {
            id<MTLTexture> array =
                make(MTLTextureType2DMultisampleArray, num_layers, sample_format, 1, true);
            set(TextureType::ColorArray2D, array);
            set(TextureType::ColorArray1D, array);
        }
        break;
    }
    default: {
        // 2D and 2D array parents, 1D images are stored as 2D textures.
        id<MTLTexture> single = make(MTLTextureType2D, 1, sample_format, num_levels, true);
        id<MTLTexture> array =
            make(MTLTextureType2DArray, num_layers, sample_format, num_levels, true);
        set(TextureType::Color1D, single);
        set(TextureType::Color2D, single);
        set(TextureType::Color2DRect, single);
        set(TextureType::ColorArray1D, array);
        set(TextureType::ColorArray2D, array);
        const bool cube_view =
            info.type == ImageViewType::Cube || info.type == ImageViewType::CubeArray;
        if (cube_view && parent_type == MTLTextureType2DArray && num_layers >= 6) {
            set(TextureType::ColorCube,
                make(MTLTextureTypeCube, 6, sample_format, num_levels, true));
            if (num_layers % 6 == 0) {
                set(TextureType::ColorArrayCube,
                    make(MTLTextureTypeCubeArray, num_layers, sample_format, num_levels, true));
            }
        }
        break;
    }
    }

    if ((image_handle.usage & MTLTextureUsageRenderTarget) != 0) {
        MTLTextureType rt_type = parent_type;
        NSUInteger rt_layers = num_layers;
        if (parent_type == MTLTextureType3D) {
            rt_layers = 1;
        } else if (parent_type == MTLTextureType2D && num_layers == 1) {
            rt_type = MTLTextureType2D;
        }
        render_target = make(rt_type, rt_layers, pixel_format, 1, false);
    }
}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo& info,
                     const ImageViewInfo& view_info, GPUVAddr gpu_addr_)
    : VideoCommon::ImageViewBase{info, view_info, gpu_addr_},
      buffer_size{VideoCommon::CalculateGuestSizeInBytes(info)} {}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::NullImageViewParams& params)
    : VideoCommon::ImageViewBase{params}, view_format{MTLPixelFormatRGBA8Unorm} {
    id<MTLDevice> device = runtime.device.GetMTLDevice();
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = 1;
    descriptor.height = 1;
    descriptor.arrayLength = 6;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    null_array = [device newTextureWithDescriptor:descriptor];
    descriptor.textureType = MTLTextureType3D;
    descriptor.arrayLength = 1;
    null_3d = [device newTextureWithDescriptor:descriptor];
    if (!null_array || !null_3d) {
        LOG_ERROR(Render_Metal, "Failed to create the null image view");
        return;
    }

    // Private textures start undefined, clear them to transparent black.
    const StagingBufferRef zeros = runtime.staging_pool.Request(4, MemoryUsage::Upload);
    std::memset(zeros.mapped_span.data(), 0, 4);
    id<MTLBlitCommandEncoder> blit = runtime.scheduler.BlitEncoder(Scheduler::PassEnd::ImageBlit);
    for (NSUInteger slice = 0; slice < 6; ++slice) {
        [blit copyFromBuffer:zeros.buffer
                   sourceOffset:zeros.offset
              sourceBytesPerRow:4
            sourceBytesPerImage:4
                     sourceSize:MTLSizeMake(1, 1, 1)
                      toTexture:null_array
               destinationSlice:slice
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit copyFromBuffer:zeros.buffer
               sourceOffset:zeros.offset
          sourceBytesPerRow:4
        sourceBytesPerImage:4
                 sourceSize:MTLSizeMake(1, 1, 1)
                  toTexture:null_3d
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];

    const auto view = [&](MTLTextureType type, NSUInteger layers) {
        return [null_array newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                             textureType:type
                                                  levels:NSMakeRange(0, 1)
                                                  slices:NSMakeRange(0, layers)];
    };
    using Shader::TextureType;
    id<MTLTexture> single = view(MTLTextureType2D, 1);
    id<MTLTexture> array = view(MTLTextureType2DArray, 1);
    views[static_cast<size_t>(TextureType::Color1D)] = single;
    views[static_cast<size_t>(TextureType::Color2D)] = single;
    views[static_cast<size_t>(TextureType::Color2DRect)] = single;
    views[static_cast<size_t>(TextureType::ColorArray1D)] = array;
    views[static_cast<size_t>(TextureType::ColorArray2D)] = array;
    views[static_cast<size_t>(TextureType::ColorCube)] = view(MTLTextureTypeCube, 6);
    views[static_cast<size_t>(TextureType::ColorArrayCube)] = view(MTLTextureTypeCubeArray, 6);
    views[static_cast<size_t>(TextureType::Color3D)] = null_3d;
}

ImageView::~ImageView() = default;

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc) {
    using Tegra::Texture::TextureFilter;
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = tsc.min_filter == TextureFilter::Linear ? MTLSamplerMinMagFilterLinear
                                                                   : MTLSamplerMinMagFilterNearest;
    descriptor.magFilter = tsc.mag_filter == TextureFilter::Linear ? MTLSamplerMinMagFilterLinear
                                                                   : MTLSamplerMinMagFilterNearest;
    switch (tsc.mipmap_filter) {
    case TextureMipmapFilter::None:
        descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
        break;
    case TextureMipmapFilter::Nearest:
        descriptor.mipFilter = MTLSamplerMipFilterNearest;
        break;
    case TextureMipmapFilter::Linear:
        descriptor.mipFilter = MTLSamplerMipFilterLinear;
        break;
    }
    descriptor.sAddressMode = WrapMode(tsc.wrap_u, tsc.mag_filter);
    descriptor.tAddressMode = WrapMode(tsc.wrap_v, tsc.mag_filter);
    descriptor.rAddressMode = WrapMode(tsc.wrap_p, tsc.mag_filter);
    descriptor.maxAnisotropy = static_cast<NSUInteger>(
        std::clamp(std::lround(tsc.MaxAnisotropy()), 1L, 16L));
    if (tsc.mipmap_filter != TextureMipmapFilter::None) {
        descriptor.lodMinClamp = tsc.MinLod();
        descriptor.lodMaxClamp = (std::max)(tsc.MaxLod(), tsc.MinLod());
    } else {
        descriptor.lodMinClamp = 0.0f;
        descriptor.lodMaxClamp = 0.25f;
    }
    if (tsc.depth_compare_enabled != 0) {
        descriptor.compareFunction = CompareFunction(tsc.depth_compare_func);
    }
    descriptor.borderColor = BorderColor(tsc.BorderColor(), approximated_border_color);
    descriptor.normalizedCoordinates = YES;
    descriptor.supportArgumentBuffers = YES;
    lod_bias = tsc.LodBias();

    if (tsc.reduction_filter != Tegra::Texture::SamplerReduction::WeightedAverage) {
        static bool logged = false;
        if (!logged) {
            LOG_WARNING(Render_Metal, "Min/max sampler reduction is not supported by Metal");
            logged = true;
        }
    }
    sampler = [runtime.device.GetMTLDevice() newSamplerStateWithDescriptor:descriptor];
    if (!sampler) {
        LOG_ERROR(Render_Metal, "Failed to create a sampler state");
    }
}

Framebuffer::Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const RenderTargets& key)
    : width{key.size.width}, height{key.size.height} {
    // The descriptor getters return autoreleased objects, keep them out of the caller's pool.
    @autoreleasepool {
        Setup(color_buffers, depth_buffer);
    }
}

void Framebuffer::Setup(std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer) {
    descriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    color_formats.fill(MTLPixelFormatInvalid);
    NSUInteger layers = 1;
    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const view = color_buffers[index];
        if (!view || !view->RenderTarget()) {
            continue;
        }
        MTLRenderPassColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
        attachment.texture = view->RenderTarget();
        attachment.loadAction = MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionStore;
        color_formats[index] = view->Format();
        samples = view->Samples();
        layers = (std::max<NSUInteger>)(layers, view->range.extent.layers);
    }
    if (depth_buffer && depth_buffer->RenderTarget()) {
        id<MTLTexture> target = depth_buffer->RenderTarget();
        depth_image = depth_buffer->ImageHandle();
        depth_format = depth_buffer->Format();
        samples = depth_buffer->Samples();
        layers = (std::max<NSUInteger>)(layers, depth_buffer->range.extent.layers);
        // The scheduler finalizes the store actions when the pass ends, see Descriptor().
        if (HasDepthAspect(depth_format)) {
            descriptor.depthAttachment.texture = target;
            descriptor.depthAttachment.loadAction = MTLLoadActionLoad;
            descriptor.depthAttachment.storeAction = MTLStoreActionUnknown;
        }
        if (HasStencilAspect(depth_format)) {
            has_stencil = true;
            descriptor.stencilAttachment.texture = target;
            descriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
            descriptor.stencilAttachment.storeAction = MTLStoreActionUnknown;
        }
    }
    if (layers > 1) {
        descriptor.renderTargetArrayLength = layers;
    }
    descriptor.renderTargetWidth = width;
    descriptor.renderTargetHeight = height;
    descriptor.defaultRasterSampleCount = samples;
}

Framebuffer::~Framebuffer() = default;

} // namespace Metal

namespace VideoCommon {
template class VideoCommon::TextureCache<Metal::TextureCacheParams>;
}
