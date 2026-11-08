// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_shader.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/textures/texture.h"

namespace Metal {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewInOut;

/// Translated guest graphics shaders, shared by every pipeline that uses the same programs and
/// translation state.
struct GraphicsProgram {
    u64 vertex_hash = 0;
    u64 fragment_hash = 0;
    std::array<Shader::Info, VideoCommon::NUM_STAGES> infos{};
    std::array<bool, VideoCommon::NUM_STAGES> enabled{};
    std::array<u32, VideoCommon::NUM_STAGES> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    /// The vertex stage writes Layer or ViewportIndex, the pipeline needs a topology class.
    bool layered = false;
    /// (geometry slot, vertex slot) pairs of the constant buffers of a folded geometry shader.
    std::vector<std::pair<u32, u32>> geometry_cbuf_remap;
};

struct ComputeProgram {
    ComputePipelineKey key{};
    Shader::Info info{};
    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};
};

namespace {

/// Guest stage indices with a native Metal equivalent.
constexpr size_t VERTEX_STAGE = 0;
/// Guest geometry stage, only reached through programs folded into the vertex stage.
constexpr size_t GEOMETRY_STAGE = 3;
constexpr size_t FRAGMENT_STAGE = 4;
constexpr std::array<size_t, 2> RENDER_STAGES{VERTEX_STAGE, FRAGMENT_STAGE};

// MAXWELL3D_REG_INDEX names Maxwell3D unqualified.
using Tegra::Engines::Maxwell3D;
#define OFF(field_name) MAXWELL3D_REG_INDEX(field_name)
#define NUM(field_name) (sizeof(Maxwell::field_name) / (sizeof(u32)))

/// Registers the Metal dirty flags, with the register ranges of the Vulkan state tracker. The
/// second table is used where the common flags already occupy the first one.
void SetupMetalDirtyFlags(Tegra::Engines::Maxwell3D::DirtyState::Tables& tables) {
    using VideoCommon::Dirty::FillBlock;
    FillBlock(tables[0], OFF(viewport_transform), NUM(viewport_transform), Dirty::Viewports);
    FillBlock(tables[0], OFF(viewports), NUM(viewports), Dirty::Viewports);
    FillBlock(tables[1], OFF(surface_clip), NUM(surface_clip), Dirty::Viewports);
    tables[0][OFF(viewport_scale_offset_enabled)] = Dirty::Viewports;
    tables[1][OFF(window_origin)] = Dirty::Viewports;
    FillBlock(tables[0], OFF(scissor_test), NUM(scissor_test), Dirty::Scissors);

    tables[0][OFF(depth_bias)] = Dirty::RasterState;
    tables[0][OFF(depth_bias_clamp)] = Dirty::RasterState;
    tables[0][OFF(slope_scale_depth_bias)] = Dirty::RasterState;
    FillBlock(tables[0], OFF(blend_color), NUM(blend_color), Dirty::RasterState);
    tables[0][OFF(gl_cull_face)] = Dirty::RasterState;
    tables[0][OFF(gl_cull_test_enabled)] = Dirty::RasterState;
    tables[0][OFF(gl_front_face)] = Dirty::RasterState;
    tables[0][OFF(polygon_mode_front)] = Dirty::RasterState;
    tables[0][OFF(viewport_clip_control)] = Dirty::RasterState;
    tables[0][OFF(polygon_offset_point_enable)] = Dirty::RasterState;
    tables[0][OFF(polygon_offset_line_enable)] = Dirty::RasterState;
    tables[0][OFF(polygon_offset_fill_enable)] = Dirty::RasterState;

    tables[0][OFF(depth_test_enable)] = Dirty::DepthStencil;
    tables[0][OFF(depth_write_enabled)] = Dirty::DepthStencil;
    tables[0][OFF(depth_test_func)] = Dirty::DepthStencil;
    tables[0][OFF(depth_bounds_enable)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_enable)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_two_side_enable)] = Dirty::DepthStencil;
    FillBlock(tables[0], OFF(stencil_front_op), NUM(stencil_front_op), Dirty::DepthStencil);
    FillBlock(tables[0], OFF(stencil_back_op), NUM(stencil_back_op), Dirty::DepthStencil);
    tables[0][OFF(stencil_front_ref)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_front_mask)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_front_func_mask)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_back_ref)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_back_mask)] = Dirty::DepthStencil;
    tables[0][OFF(stencil_back_func_mask)] = Dirty::DepthStencil;

    tables[0][OFF(rasterize_enable)] = Dirty::PipelineState;
    tables[0][OFF(anti_alias_alpha_control)] = Dirty::PipelineState;
    tables[0][OFF(color_mask_common)] = Dirty::PipelineState;
    tables[0][OFF(blend_per_target_enabled)] = Dirty::PipelineState;
    FillBlock(tables[0], OFF(color_mask), NUM(color_mask), Dirty::PipelineState);
    FillBlock(tables[0], OFF(blend), NUM(blend), Dirty::PipelineState);
    FillBlock(tables[0], OFF(blend_per_target), NUM(blend_per_target), Dirty::PipelineState);
    FillBlock(tables[1], OFF(vertex_attrib_format), NUM(vertex_attrib_format),
              Dirty::PipelineState);
    // Strides set the common VertexBuffers flag, instancing lives in the divisor words.
    static constexpr size_t divisor_offset = 3;
    for (size_t i = 0; i < Maxwell::NumVertexArrays; ++i) {
        tables[0][OFF(vertex_stream_instances) + i] = Dirty::PipelineState;
        tables[0][OFF(vertex_streams) + i * NUM(vertex_streams[0]) + divisor_offset] =
            Dirty::PipelineState;
    }
}

#undef OFF
#undef NUM

/// Stride and instancing of a vertex stream, the only parts of it baked into pipelines.
VertexBufferKey MakeVertexBufferKey(const Maxwell& regs, size_t index) {
    const auto& stream = regs.vertex_streams[index];
    VertexBufferKey out{};
    out.stride = stream.stride;
    if (out.stride == 0) {
        out.step_function = MTLVertexStepFunctionConstant;
        out.step_rate = 0;
    } else if (regs.vertex_stream_instances.IsInstancingEnabled(index)) {
        out.step_function = MTLVertexStepFunctionPerInstance;
        out.step_rate = (std::max)(stream.frequency, 1U);
    } else {
        out.step_function = MTLVertexStepFunctionPerVertex;
        out.step_rate = 1;
    }
    return out;
}

/// Identifies a resource list for residency, the lists are rebuilt in binding order.
template <typename List>
u64 HashResourceList(const List& list) {
    u64 hash = 0x9E3779B97F4A7C15ULL;
    for (const auto& item : list) {
        const u64 value = reinterpret_cast<u64>((__bridge void*)item);
        hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
}

constexpr u32 UTIL_KIND_FLOAT = 0;
constexpr u32 UTIL_KIND_UINT = 1;
constexpr u32 UTIL_KIND_SINT = 2;
constexpr u32 UTIL_KIND_DEPTH_STENCIL = 3;
constexpr u32 UTIL_KIND_BLIT = 4;

constexpr u64 DEPTH_STENCIL_CUSTOM_BIT = u64{1} << 63;

/// Layout shared with the utility shaders below.
struct UtilParams {
    std::array<f32, 4> color;
    std::array<u32, 4> ucolor;
    std::array<f32, 4> rect;
    std::array<f32, 4> uv;
    f32 depth;
    u32 base_layer;
    std::array<u32, 2> padding;
};
static_assert(sizeof(UtilParams) == 80);

constexpr const char* UTILITY_SOURCE = R"(#include <metal_stdlib>
using namespace metal;
struct UtilParams {
    float4 color;
    uint4 ucolor;
    float4 rect;
    float4 uv;
    float depth;
    uint base_layer;
    uint2 padding;
};
struct ClearOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};
vertex ClearOut eden_clear_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                              constant UtilParams& p [[buffer(0)]]) {
    const float2 uv = float2(float((vid << 1) & 2), float(vid & 2));
    ClearOut out;
    out.position = float4(uv * 2.0 - 1.0, p.depth, 1.0);
    out.layer = p.base_layer + iid;
    return out;
}
struct BlitOut {
    float4 position [[position]];
    float2 uv;
};
vertex BlitOut eden_blit_vs(uint vid [[vertex_id]], constant UtilParams& p [[buffer(0)]]) {
    const float2 t = float2(float(vid & 1), float(vid >> 1));
    BlitOut out;
    out.position = float4(mix(p.rect.xy, p.rect.zw, t), 0.0, 1.0);
    out.uv = mix(p.uv.xy, p.uv.zw, t);
    return out;
}
fragment float4 eden_blit_fs(BlitOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                             sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}
)";

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block;
};

/// Everything the guest translation of a graphics program depends on besides the code.
struct GraphicsProgramKey {
    std::array<u64, 6> unique_hashes;
    std::array<u8, Maxwell::NumVertexAttributes> attribute_types;
    std::array<u8, Maxwell::NumRenderTargets> color_output_types;
    u32 alpha_test_ref;
    u32 point_size;
    u8 alpha_test_func;
    u8 input_topology;
    u8 is_points;
    u8 ndc_minus_one_to_one;
    u8 y_negate;
    u8 early_z;
    u8 dual_source_blend;
    u8 padding;

    bool operator==(const GraphicsProgramKey& rhs) const noexcept {
        return std::memcmp(this, &rhs, sizeof *this) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsProgramKey>);

struct ComputeProgramKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    bool operator==(const ComputeProgramKey& rhs) const noexcept {
        return std::memcmp(this, &rhs, sizeof *this) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<ComputeProgramKey>);

struct BytesHash {
    template <typename T>
    size_t operator()(const T& value) const noexcept {
        return static_cast<size_t>(
            Common::CityHash64(reinterpret_cast<const char*>(&value), sizeof(T)));
    }
};

template <typename T>
u64 HashBytes(const T& value) {
    return Common::CityHash64(reinterpret_cast<const char*>(&value), sizeof(T));
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    return Shader::CompareFunction::Always;
}

Shader::AttributeType AttributeTypeOf(const Maxwell::VertexAttribute& attribute) {
    using Type = Maxwell::VertexAttribute::Type;
    if (attribute.constant) {
        return Shader::AttributeType::Disabled;
    }
    switch (attribute.type) {
    case Type::SNorm:
    case Type::UNorm:
    case Type::Float:
        return Shader::AttributeType::Float;
    case Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    default:
        return Shader::AttributeType::Disabled;
    }
}

Shader::InputTopology InputTopologyOf(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::InputTopology::Points;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::InputTopology::Lines;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return Shader::InputTopology::LinesAdjacency;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        return Shader::InputTopology::TrianglesAdjacency;
    default:
        return Shader::InputTopology::Triangles;
    }
}

bool IsDualSourceBlendFactor(Maxwell::Blend::Factor factor) {
    using F = Maxwell::Blend::Factor;
    switch (factor) {
    case F::Source1Color_D3D:
    case F::OneMinusSource1Color_D3D:
    case F::Source1Alpha_D3D:
    case F::OneMinusSource1Alpha_D3D:
    case F::Source1Color_GL:
    case F::OneMinusSource1Color_GL:
    case F::Source1Alpha_GL:
    case F::OneMinusSource1Alpha_GL:
        return true;
    default:
        return false;
    }
}

bool UsesDualSourceBlend(const Maxwell& regs) {
    if (!regs.blend.enable[0]) {
        return false;
    }
    const auto uses = [](const auto& blend) {
        return IsDualSourceBlendFactor(blend.color_source) ||
               IsDualSourceBlendFactor(blend.color_dest) ||
               IsDualSourceBlendFactor(blend.alpha_source) ||
               IsDualSourceBlendFactor(blend.alpha_dest);
    };
    return regs.blend_per_target_enabled ? uses(regs.blend_per_target[0]) : uses(regs.blend);
}

/// Converts both the D3D (1-8) and GL (0x200-0x207) encodings, the result matches the order of
/// MTLCompareFunction.
u32 PackComparisonOp(Maxwell::ComparisonOp op) {
    const u32 value = static_cast<u32>(op);
    const u32 packed = value - (value >= 0x200 ? 0x200 : 1);
    return (std::min)(packed, 7U);
}

MTLCompareFunction CompareFunction(Maxwell::ComparisonOp op) {
    return static_cast<MTLCompareFunction>(PackComparisonOp(op));
}

u32 PackStencilOp(Maxwell::StencilOp::Op op) {
    using Op = Maxwell::StencilOp::Op;
    switch (op) {
    case Op::Keep_D3D:
    case Op::Keep_GL:
        return MTLStencilOperationKeep;
    case Op::Zero_D3D:
    case Op::Zero_GL:
        return MTLStencilOperationZero;
    case Op::Replace_D3D:
    case Op::Replace_GL:
        return MTLStencilOperationReplace;
    case Op::IncrSaturate_D3D:
    case Op::IncrSaturate_GL:
        return MTLStencilOperationIncrementClamp;
    case Op::DecrSaturate_D3D:
    case Op::DecrSaturate_GL:
        return MTLStencilOperationDecrementClamp;
    case Op::Invert_D3D:
    case Op::Invert_GL:
        return MTLStencilOperationInvert;
    case Op::Incr_D3D:
    case Op::Incr_GL:
        return MTLStencilOperationIncrementWrap;
    case Op::Decr_D3D:
    case Op::Decr_GL:
        return MTLStencilOperationDecrementWrap;
    }
    return MTLStencilOperationKeep;
}

MTLBlendOperation BlendOperation(Maxwell::Blend::Equation equation) {
    using E = Maxwell::Blend::Equation;
    switch (equation) {
    case E::Add_D3D:
    case E::Add_GL:
        return MTLBlendOperationAdd;
    case E::Subtract_D3D:
    case E::Subtract_GL:
        return MTLBlendOperationSubtract;
    case E::ReverseSubtract_D3D:
    case E::ReverseSubtract_GL:
        return MTLBlendOperationReverseSubtract;
    case E::Min_D3D:
    case E::Min_GL:
        return MTLBlendOperationMin;
    case E::Max_D3D:
    case E::Max_GL:
        return MTLBlendOperationMax;
    }
    return MTLBlendOperationAdd;
}

MTLBlendFactor BlendFactor(Maxwell::Blend::Factor factor) {
    using F = Maxwell::Blend::Factor;
    switch (factor) {
    case F::Zero_D3D:
    case F::Zero_GL:
        return MTLBlendFactorZero;
    case F::One_D3D:
    case F::One_GL:
        return MTLBlendFactorOne;
    case F::SourceColor_D3D:
    case F::SourceColor_GL:
        return MTLBlendFactorSourceColor;
    case F::OneMinusSourceColor_D3D:
    case F::OneMinusSourceColor_GL:
        return MTLBlendFactorOneMinusSourceColor;
    case F::SourceAlpha_D3D:
    case F::SourceAlpha_GL:
    case F::BothSourceAlpha_D3D:
        return MTLBlendFactorSourceAlpha;
    case F::OneMinusSourceAlpha_D3D:
    case F::OneMinusSourceAlpha_GL:
    case F::OneMinusBothSourceAlpha_D3D:
        return MTLBlendFactorOneMinusSourceAlpha;
    case F::DestAlpha_D3D:
    case F::DestAlpha_GL:
        return MTLBlendFactorDestinationAlpha;
    case F::OneMinusDestAlpha_D3D:
    case F::OneMinusDestAlpha_GL:
        return MTLBlendFactorOneMinusDestinationAlpha;
    case F::DestColor_D3D:
    case F::DestColor_GL:
        return MTLBlendFactorDestinationColor;
    case F::OneMinusDestColor_D3D:
    case F::OneMinusDestColor_GL:
        return MTLBlendFactorOneMinusDestinationColor;
    case F::SourceAlphaSaturate_D3D:
    case F::SourceAlphaSaturate_GL:
        return MTLBlendFactorSourceAlphaSaturated;
    case F::BlendFactor_D3D:
    case F::ConstantColor_GL:
        return MTLBlendFactorBlendColor;
    case F::OneMinusBlendFactor_D3D:
    case F::OneMinusConstantColor_GL:
        return MTLBlendFactorOneMinusBlendColor;
    case F::ConstantAlpha_GL:
        return MTLBlendFactorBlendAlpha;
    case F::OneMinusConstantAlpha_GL:
        return MTLBlendFactorOneMinusBlendAlpha;
    case F::Source1Color_D3D:
    case F::Source1Color_GL:
        return MTLBlendFactorSource1Color;
    case F::OneMinusSource1Color_D3D:
    case F::OneMinusSource1Color_GL:
        return MTLBlendFactorOneMinusSource1Color;
    case F::Source1Alpha_D3D:
    case F::Source1Alpha_GL:
        return MTLBlendFactorSource1Alpha;
    case F::OneMinusSource1Alpha_D3D:
    case F::OneMinusSource1Alpha_GL:
        return MTLBlendFactorOneMinusSource1Alpha;
    }
    return MTLBlendFactorOne;
}

/// Scaled attributes are read as integers, the shader recompiler converts them to float.
MTLVertexFormat VertexFormat(Maxwell::VertexAttribute::Type type,
                             Maxwell::VertexAttribute::Size size) {
    using T = Maxwell::VertexAttribute::Type;
    using S = Maxwell::VertexAttribute::Size;
    if (type == T::UScaled) {
        type = T::UInt;
    } else if (type == T::SScaled) {
        type = T::SInt;
    }
    switch (type) {
    case T::UNorm:
        switch (size) {
        case S::Size_R8:
        case S::Size_A8:
            return MTLVertexFormatUCharNormalized;
        case S::Size_R8_G8:
        case S::Size_G8_R8:
            return MTLVertexFormatUChar2Normalized;
        case S::Size_R8_G8_B8:
            return MTLVertexFormatUChar3Normalized;
        case S::Size_R8_G8_B8_A8:
        case S::Size_X8_B8_G8_R8:
            return MTLVertexFormatUChar4Normalized;
        case S::Size_R16:
            return MTLVertexFormatUShortNormalized;
        case S::Size_R16_G16:
            return MTLVertexFormatUShort2Normalized;
        case S::Size_R16_G16_B16:
            return MTLVertexFormatUShort3Normalized;
        case S::Size_R16_G16_B16_A16:
            return MTLVertexFormatUShort4Normalized;
        case S::Size_A2_B10_G10_R10:
            return MTLVertexFormatUInt1010102Normalized;
        default:
            break;
        }
        break;
    case T::SNorm:
        switch (size) {
        case S::Size_R8:
        case S::Size_A8:
            return MTLVertexFormatCharNormalized;
        case S::Size_R8_G8:
        case S::Size_G8_R8:
            return MTLVertexFormatChar2Normalized;
        case S::Size_R8_G8_B8:
            return MTLVertexFormatChar3Normalized;
        case S::Size_R8_G8_B8_A8:
        case S::Size_X8_B8_G8_R8:
            return MTLVertexFormatChar4Normalized;
        case S::Size_R16:
            return MTLVertexFormatShortNormalized;
        case S::Size_R16_G16:
            return MTLVertexFormatShort2Normalized;
        case S::Size_R16_G16_B16:
            return MTLVertexFormatShort3Normalized;
        case S::Size_R16_G16_B16_A16:
            return MTLVertexFormatShort4Normalized;
        case S::Size_A2_B10_G10_R10:
            return MTLVertexFormatInt1010102Normalized;
        default:
            break;
        }
        break;
    case T::UInt:
        switch (size) {
        case S::Size_R8:
        case S::Size_A8:
            return MTLVertexFormatUChar;
        case S::Size_R8_G8:
        case S::Size_G8_R8:
            return MTLVertexFormatUChar2;
        case S::Size_R8_G8_B8:
            return MTLVertexFormatUChar3;
        case S::Size_R8_G8_B8_A8:
        case S::Size_X8_B8_G8_R8:
            return MTLVertexFormatUChar4;
        case S::Size_R16:
            return MTLVertexFormatUShort;
        case S::Size_R16_G16:
            return MTLVertexFormatUShort2;
        case S::Size_R16_G16_B16:
            return MTLVertexFormatUShort3;
        case S::Size_R16_G16_B16_A16:
            return MTLVertexFormatUShort4;
        case S::Size_R32:
            return MTLVertexFormatUInt;
        case S::Size_R32_G32:
            return MTLVertexFormatUInt2;
        case S::Size_R32_G32_B32:
            return MTLVertexFormatUInt3;
        case S::Size_R32_G32_B32_A32:
            return MTLVertexFormatUInt4;
        default:
            break;
        }
        break;
    case T::SInt:
        switch (size) {
        case S::Size_R8:
        case S::Size_A8:
            return MTLVertexFormatChar;
        case S::Size_R8_G8:
        case S::Size_G8_R8:
            return MTLVertexFormatChar2;
        case S::Size_R8_G8_B8:
            return MTLVertexFormatChar3;
        case S::Size_R8_G8_B8_A8:
        case S::Size_X8_B8_G8_R8:
            return MTLVertexFormatChar4;
        case S::Size_R16:
            return MTLVertexFormatShort;
        case S::Size_R16_G16:
            return MTLVertexFormatShort2;
        case S::Size_R16_G16_B16:
            return MTLVertexFormatShort3;
        case S::Size_R16_G16_B16_A16:
            return MTLVertexFormatShort4;
        case S::Size_R32:
            return MTLVertexFormatInt;
        case S::Size_R32_G32:
            return MTLVertexFormatInt2;
        case S::Size_R32_G32_B32:
            return MTLVertexFormatInt3;
        case S::Size_R32_G32_B32_A32:
            return MTLVertexFormatInt4;
        default:
            break;
        }
        break;
    case T::Float:
        switch (size) {
        case S::Size_R16:
            return MTLVertexFormatHalf;
        case S::Size_R16_G16:
            return MTLVertexFormatHalf2;
        case S::Size_R16_G16_B16:
            return MTLVertexFormatHalf3;
        case S::Size_R16_G16_B16_A16:
            return MTLVertexFormatHalf4;
        case S::Size_R32:
            return MTLVertexFormatFloat;
        case S::Size_R32_G32:
            return MTLVertexFormatFloat2;
        case S::Size_R32_G32_B32:
            return MTLVertexFormatFloat3;
        case S::Size_R32_G32_B32_A32:
            return MTLVertexFormatFloat4;
        default:
            break;
        }
        break;
    default:
        break;
    }
    LOG_WARNING(Render_Metal, "Unsupported vertex attribute type={} size={}", type, size);
    return MTLVertexFormatInvalid;
}

std::optional<MTLPrimitiveType> PrimitiveType(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return MTLPrimitiveTypePoint;
    case Maxwell::PrimitiveTopology::Lines:
        return MTLPrimitiveTypeLine;
    case Maxwell::PrimitiveTopology::LineStrip:
    case Maxwell::PrimitiveTopology::LineLoop:
        return MTLPrimitiveTypeLineStrip;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
        return MTLPrimitiveTypeTriangle;
    case Maxwell::PrimitiveTopology::TriangleStrip:
        return MTLPrimitiveTypeTriangleStrip;
    default:
        return std::nullopt;
    }
}

MTLPrimitiveTopologyClass TopologyClass(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return MTLPrimitiveTopologyClassPoint;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        return MTLPrimitiveTopologyClassLine;
    default:
        return MTLPrimitiveTopologyClassTriangle;
    }
}

/// Topologies Metal can't draw directly, they are rewritten as indexed lists.
bool NeedsIndexConversion(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
        return true;
    default:
        return false;
    }
}

template <typename Read>
void GenerateIndices(Maxwell::PrimitiveTopology topology, u32 count, Read&& read,
                     std::vector<u32>& out) {
    out.clear();
    switch (topology) {
    case Maxwell::PrimitiveTopology::Quads:
        for (u32 quad = 0; quad < count / 4; ++quad) {
            const u32 v = quad * 4;
            for (const u32 offset : {0U, 1U, 2U, 0U, 2U, 3U}) {
                out.push_back(read(v + offset));
            }
        }
        break;
    case Maxwell::PrimitiveTopology::QuadStrip:
        if (count >= 4) {
            for (u32 quad = 0; quad < (count - 2) / 2; ++quad) {
                const u32 v = quad * 2;
                for (const u32 offset : {0U, 1U, 3U, 0U, 3U, 2U}) {
                    out.push_back(read(v + offset));
                }
            }
        }
        break;
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
        for (u32 i = 1; i + 1 < count; ++i) {
            out.push_back(read(0));
            out.push_back(read(i));
            out.push_back(read(i + 1));
        }
        break;
    case Maxwell::PrimitiveTopology::LineLoop:
        for (u32 i = 0; i < count; ++i) {
            out.push_back(read(i));
        }
        if (count > 1) {
            out.push_back(read(0));
        }
        break;
    default:
        break;
    }
}

bool HasDepthAspect(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth16Unorm:
    case MTLPixelFormatDepth32Float:
    case MTLPixelFormatDepth32Float_Stencil8:
#if TARGET_OS_OSX
    case MTLPixelFormatDepth24Unorm_Stencil8:
#endif
        return true;
    default:
        return false;
    }
}

MTLViewport GetViewport(const Maxwell& regs, size_t index) {
    const auto& src = regs.viewport_transform[index];
    const f32 x = src.translate_x - src.scale_x;
    const f32 width = src.scale_x * 2.0f;
    f32 y = src.translate_y - src.scale_y;
    f32 height = src.scale_y * 2.0f;
    if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
        const f32 clip_height = static_cast<f32>(regs.surface_clip.height);
        if (src.scale_y < 0.0f && y >= clip_height) {
            y = clip_height - y;
        } else {
            y += clip_height;
        }
        height = -height;
    }
    if (src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
        y += height;
        height = -height;
    }
    const f32 reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    return MTLViewport{
        .originX = x,
        .originY = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .znear = std::clamp(src.translate_z - src.scale_z * reduce_z, 0.0f, 1.0f),
        .zfar = std::clamp(src.translate_z + src.scale_z, 0.0f, 1.0f),
    };
}

/// Metal requires scissor rectangles to lie inside the render targets.
MTLScissorRect ClampScissor(s64 x, s64 y, s64 width, s64 height, u32 fb_width, u32 fb_height) {
    const s64 x0 = std::clamp<s64>(x, 0, fb_width);
    const s64 y0 = std::clamp<s64>(y, 0, fb_height);
    const s64 x1 = std::clamp<s64>(x + width, x0, fb_width);
    const s64 y1 = std::clamp<s64>(y + height, y0, fb_height);
    return MTLScissorRect{
        .x = static_cast<NSUInteger>(x0),
        .y = static_cast<NSUInteger>(y0),
        .width = static_cast<NSUInteger>(x1 - x0),
        .height = static_cast<NSUInteger>(y1 - y0),
    };
}

MTLScissorRect GetScissor(const Maxwell& regs, size_t index, u32 fb_width, u32 fb_height) {
    const auto& src = regs.scissor_test[index];
    if (!src.enable) {
        return ClampScissor(0, 0, fb_width, fb_height, fb_width, fb_height);
    }
    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s64 clip_height = regs.surface_clip.height;
    const s64 min_y = lower_left ? clip_height - src.max_y : s64{src.min_y};
    const s64 max_y = lower_left ? clip_height - src.min_y : s64{src.max_y};
    return ClampScissor(src.min_x, min_y, s64{src.max_x} - src.min_x, max_y - min_y, fb_width,
                        fb_height);
}

std::pair<u32, u32> TexturePair(u32 raw, bool via_header_index) {
    if (via_header_index) {
        return {raw, raw};
    }
    const Tegra::Texture::TextureHandle handle{raw};
    return {handle.tic_id, handle.tsc_id};
}

/// Reads the texture handle of a descriptor element, cbuf_address returns the base address of
/// a constant buffer. Mirrors the Vulkan graphics pipeline.
template <typename Desc, typename CbufAddress>
std::pair<u32, u32> ReadHandle(Tegra::MemoryManager& gpu_memory, const Desc& desc, u32 index,
                               bool via_header_index, CbufAddress&& cbuf_address) {
    constexpr bool is_texture = std::is_same_v<Desc, Shader::TextureDescriptor> ||
                                std::is_same_v<Desc, Shader::TextureBufferDescriptor>;
    if (desc.cbuf_index == Shader::ATTRIBUTE_HANDLE_CBUF_INDEX) {
        static bool logged = false;
        if (!logged) {
            LOG_WARNING(Render_Metal, "Texture handles carried in vertex attributes are not "
                                      "implemented");
            logged = true;
        }
        return TexturePair(0, via_header_index);
    }
    const auto read_secondary = [&](u32 index_offset) -> u32 {
        if constexpr (is_texture) {
            if constexpr (requires { desc.secondary_is_storage; }) {
                if (desc.secondary_cbuf_index == 0xFFFEU) {
                    return desc.secondary_cbuf_offset;
                }
            }
            const GPUVAddr base = cbuf_address(desc.secondary_cbuf_index) +
                                  desc.secondary_cbuf_offset;
            if constexpr (requires { desc.secondary_is_storage; }) {
                if (desc.secondary_is_storage) {
                    const GPUVAddr ptr = gpu_memory.Read<u64>(base);
                    return gpu_memory.Read<u32>(
                        static_cast<GPUVAddr>(static_cast<s64>(ptr) + desc.secondary_storage_offset));
                }
            }
            return gpu_memory.Read<u32>(base + index_offset);
        }
        return 0;
    };
    const u32 index_offset = index << desc.size_shift;
    const GPUVAddr cbuf_base = cbuf_address(desc.cbuf_index);
    if constexpr (requires { desc.is_storage; }) {
        if (desc.is_storage) {
            const s64 handle_offset = s64{desc.storage_offset} + s64{index} * desc.storage_stride;
            const GPUVAddr ptr = gpu_memory.Read<u64>(cbuf_base + desc.cbuf_offset);
            u32 raw = gpu_memory.Read<u32>(
                static_cast<GPUVAddr>(static_cast<s64>(ptr) + handle_offset));
            if constexpr (is_texture) {
                raw <<= desc.shift_left;
                if (desc.has_secondary) {
                    raw |= read_secondary(0) << desc.secondary_shift_left;
                }
            }
            return TexturePair(raw, via_header_index);
        }
    }
    const GPUVAddr addr = cbuf_base + desc.cbuf_offset + index_offset;
    if constexpr (is_texture) {
        if (desc.has_secondary) {
            const u32 lhs = gpu_memory.Read<u32>(addr) << desc.shift_left;
            const u32 rhs = read_secondary(index_offset) << desc.secondary_shift_left;
            return TexturePair(lhs | rhs, via_header_index);
        }
    }
    return TexturePair(gpu_memory.Read<u32>(addr), via_header_index);
}

void WarnOnce(bool& flag, std::string_view message) {
    if (!flag) {
        LOG_WARNING(Render_Metal, "{}", message);
        flag = true;
    }
}

} // Anonymous namespace

/// Translates guest Maxwell programs to SPIR-V and registers them in the pipeline cache.
class ShaderFrontend final : public VideoCommon::ShaderCache {
public:
    explicit ShaderFrontend(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                            PipelineCache& pipeline_cache_)
        : VideoCommon::ShaderCache{device_memory_}, pipeline_cache{pipeline_cache_} {}

    const GraphicsProgram* CurrentGraphicsProgram() {
        if (!RefreshStages(unique_hashes)) {
            return nullptr;
        }
        const GraphicsProgramKey key = MakeKey();
        auto [it, is_new] = graphics_programs.try_emplace(key);
        if (is_new) {
            // Failures are cached as nullptr so that broken shaders are not retranslated.
            it->second = BuildGraphics(key);
        }
        return it->second.get();
    }

    const ComputeProgram* CurrentComputeProgram() {
        const VideoCommon::ShaderInfo* const shader = ComputeShader();
        if (!shader) {
            return nullptr;
        }
        const auto& qmd = kepler_compute->launch_description;
        const ComputeProgramKey key{
            .unique_hash = shader->unique_hash,
            .shared_memory_size = static_cast<u32>(qmd.shared_alloc),
            .workgroup_size = {static_cast<u32>(qmd.block_dim_x),
                               static_cast<u32>(qmd.block_dim_y),
                               static_cast<u32>(qmd.block_dim_z)},
        };
        auto [it, is_new] = compute_programs.try_emplace(key);
        if (is_new) {
            it->second = BuildCompute(key, *shader);
        }
        return it->second.get();
    }

private:
    GraphicsProgramKey MakeKey() const {
        const auto& regs = maxwell3d->regs;
        const auto topology = maxwell3d->draw_manager.draw_state.topology;
        GraphicsProgramKey key{};
        key.unique_hashes = unique_hashes;
        for (size_t i = 0; i < Maxwell::NumVertexAttributes; ++i) {
            key.attribute_types[i] =
                static_cast<u8>(AttributeTypeOf(regs.vertex_attrib_format[i]));
        }
        for (size_t i = 0; i < Maxwell::NumRenderTargets; ++i) {
            Shader::AttributeType type = Shader::AttributeType::Float;
            if (regs.rt[i].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[i].format);
                if (VideoCore::Surface::IsPixelFormatInteger(format)) {
                    type = VideoCore::Surface::IsPixelFormatSignedInteger(format)
                               ? Shader::AttributeType::SignedInt
                               : Shader::AttributeType::UnsignedInt;
                }
            }
            key.color_output_types[i] = static_cast<u8>(type);
        }
        key.alpha_test_ref = std::bit_cast<u32>(regs.alpha_test_ref);
        key.point_size = std::bit_cast<u32>(regs.point_size);
        key.alpha_test_func = static_cast<u8>(regs.alpha_test_enabled != 0
                                                  ? MaxwellToCompareFunction(regs.alpha_test_func)
                                                  : Shader::CompareFunction::Always);
        key.input_topology = static_cast<u8>(InputTopologyOf(topology));
        key.is_points = topology == Maxwell::PrimitiveTopology::Points ? 1 : 0;
        key.ndc_minus_one_to_one = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1 : 0;
        key.y_negate = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft ? 1 : 0;
        key.early_z = regs.mandated_early_z != 0 ? 1 : 0;
        key.dual_source_blend = UsesDualSourceBlend(regs) ? 1 : 0;
        return key;
    }

    static Shader::RuntimeInfo MakeRuntimeInfo(const GraphicsProgramKey& key,
                                               const Shader::IR::Program& program,
                                               const Shader::IR::Program* previous) {
        Shader::RuntimeInfo info;
        if (previous) {
            info.previous_stage_stores = previous->info.stores;
            info.previous_stage_legacy_stores_mapping = previous->info.legacy_stores_mapping;
        } else {
            info.previous_stage_stores.mask.set();
        }
        switch (program.stage) {
        case Shader::Stage::VertexB:
            if (key.is_points) {
                info.fixed_state_point_size = std::bit_cast<f32>(key.point_size);
            }
            info.convert_depth_mode = key.ndc_minus_one_to_one != 0;
            for (size_t i = 0; i < Maxwell::NumVertexAttributes; ++i) {
                info.generic_input_types[i] =
                    static_cast<Shader::AttributeType>(key.attribute_types[i]);
            }
            break;
        case Shader::Stage::Fragment:
            info.alpha_test_func = static_cast<Shader::CompareFunction>(key.alpha_test_func);
            info.alpha_test_reference = std::bit_cast<f32>(key.alpha_test_ref);
            info.dual_source_blend = key.dual_source_blend != 0;
            for (size_t i = 0; i < Maxwell::NumRenderTargets; ++i) {
                info.color_output_types[i] =
                    static_cast<Shader::AttributeType>(key.color_output_types[i]);
            }
            break;
        default:
            break;
        }
        info.input_topology = static_cast<Shader::InputTopology>(key.input_topology);
        info.force_early_z = key.early_z != 0;
        info.y_negate = key.y_negate != 0;
        return info;
    }

    void ReportUnsupportedStage(u64 hash, std::string_view reason) {
        if (reported_stages.insert(hash).second) {
            LOG_WARNING(Render_Metal, "Skipping draws using shader {:016x}: {}", hash, reason);
        }
    }

    std::unique_ptr<GraphicsProgram> BuildGraphics(const GraphicsProgramKey& key) try {
        constexpr size_t GEOMETRY_PROGRAM = 4;
        for (const size_t index : {size_t{2}, size_t{3}}) {
            if (key.unique_hashes[index] != 0) {
                // TODO: Metal tessellation needs the vertex and control stages as compute kernels.
                ReportUnsupportedStage(key.unique_hashes[index],
                                       "tessellation is not supported by the Metal backend");
                return nullptr;
            }
        }
        if (key.unique_hashes[1] == 0) {
            return nullptr;
        }
        GraphicsEnvironments environments;
        GetGraphicsEnvironments(environments, key.unique_hashes);
        const auto envs = environments.Span();

        pools.ReleaseContents();
        const Shader::HostTranslateInfo& host_info = pipeline_cache.GetHostTranslateInfo();
        // Passthrough geometry shaders are kept unlowered so they can be folded into the vertex
        // stage, Metal has no geometry stage.
        Shader::HostTranslateInfo geometry_host_info = host_info;
        geometry_host_info.support_geometry_shader_passthrough = true;
        std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
        const bool uses_vertex_a = key.unique_hashes[0] != 0;
        Shader::Environment* vertex_b_env = nullptr;
        size_t env_index = 0;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::Environment& env = *envs[env_index++];
            const u32 cfg_offset =
                static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            const Shader::HostTranslateInfo& stage_host_info =
                index == GEOMETRY_PROGRAM ? geometry_host_info : host_info;
            if (!uses_vertex_a || index != 1) {
                programs[index] = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env,
                                                                    cfg, stage_host_info);
            } else {
                auto program_vb = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env,
                                                                    cfg, host_info);
                programs[index] =
                    Shader::Maxwell::MergeDualVertexPrograms(programs[0], program_vb, env);
            }
            if (index == 1) {
                vertex_b_env = &env;
            }
        }
        std::vector<std::pair<u32, u32>> geometry_cbuf_remap;
        if (key.unique_hashes[GEOMETRY_PROGRAM] != 0) {
            std::string_view reason{"unknown reason"};
            std::optional<Shader::IR::Program> folded = Shader::Maxwell::FoldGeometryPassthrough(
                programs[1], programs[GEOMETRY_PROGRAM], *vertex_b_env, geometry_cbuf_remap,
                &reason);
            if (!folded) {
                ReportUnsupportedStage(
                    key.unique_hashes[GEOMETRY_PROGRAM],
                    fmt::format("the geometry shader can't be folded into the vertex stage: {}",
                                reason));
                return nullptr;
            }
            programs[1] = std::move(*folded);
        }

        auto result = std::make_unique<GraphicsProgram>();
        result->geometry_cbuf_remap = std::move(geometry_cbuf_remap);
        const Shader::IR::Program* previous = nullptr;
        for (const size_t index : {size_t{1}, size_t{5}}) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::IR::Program& program = programs[index];
            const size_t stage = index - 1;
            const Shader::RuntimeInfo runtime_info = MakeRuntimeInfo(key, program, previous);
            Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);
            // Every Metal stage has its own argument buffer, bindings start at zero.
            Shader::Backend::Bindings bindings;
            const std::vector<u32> code = Shader::Backend::SPIRV::EmitSPIRV(
                pipeline_cache.GetProfile(), runtime_info, program, bindings);
            const ShaderStage mtl_stage =
                stage == VERTEX_STAGE ? ShaderStage::Vertex : ShaderStage::Fragment;
            const u64 hash = pipeline_cache.RegisterShader(mtl_stage, code);
            if (stage == VERTEX_STAGE) {
                result->vertex_hash = hash;
                result->layered = program.info.stores[Shader::IR::Attribute::Layer] ||
                                  program.info.stores[Shader::IR::Attribute::ViewportIndex];
            } else {
                result->fragment_hash = hash;
            }
            result->infos[stage] = program.info;
            result->enabled[stage] = true;
            result->enabled_uniform_buffer_masks[stage] = program.info.constant_buffer_mask;
            std::ranges::copy(program.info.constant_buffer_used_sizes,
                              result->uniform_buffer_sizes[stage].begin());
            previous = &program;
        }
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Metal, "Failed to translate graphics shaders: {}", e.what());
        return nullptr;
    }

    std::unique_ptr<ComputeProgram> BuildCompute(const ComputeProgramKey& key,
                                                 const VideoCommon::ShaderInfo& shader) try {
        const GPUVAddr program_base = kepler_compute->regs.code_loc.Address();
        const auto& qmd = kepler_compute->launch_description;
        VideoCommon::ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base,
                                            qmd.program_start};
        env.SetCachedSize(shader.size_bytes);

        pools.ReleaseContents();
        Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};
        Shader::IR::Program program = Shader::Maxwell::TranslateProgram(
            pools.inst, pools.block, env, cfg, pipeline_cache.GetHostTranslateInfo());
        const std::vector<u32> code =
            Shader::Backend::SPIRV::EmitSPIRV(pipeline_cache.GetProfile(), program);

        auto result = std::make_unique<ComputeProgram>();
        result->key = ComputePipelineKey{
            .shader = pipeline_cache.RegisterShader(ShaderStage::Compute, code),
            .shared_memory_size = key.shared_memory_size,
            .workgroup_size = key.workgroup_size,
        };
        result->info = program.info;
        std::copy_n(program.info.constant_buffer_used_sizes.begin(),
                    result->uniform_buffer_sizes.size(), result->uniform_buffer_sizes.begin());
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Metal, "Failed to translate compute shader: {}", e.what());
        return nullptr;
    }

    PipelineCache& pipeline_cache;
    ShaderPools pools;
    std::array<u64, 6> unique_hashes{};
    std::unordered_set<u64> reported_stages;
    std::unordered_map<GraphicsProgramKey, std::unique_ptr<GraphicsProgram>, BytesHash>
        graphics_programs;
    std::unordered_map<ComputeProgramKey, std::unique_ptr<ComputeProgram>, BytesHash>
        compute_programs;
};

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : VideoCommon::FenceBase{is_stubbed_}, scheduler{scheduler_} {}

void InnerFence::Queue() {
    if (is_stubbed) {
        return;
    }
    if (scheduler.IsEmpty()) {
        // Nothing was recorded since the last submission, which already covers this fence.
        // Submitting an empty command buffer would cost a kernel round trip for nothing.
        wait_tick = scheduler.CurrentTick() - 1;
        return;
    }
    wait_tick = scheduler.CurrentTick();
    scheduler.Flush();
}

bool InnerFence::IsSignaled() const {
    return is_stubbed || scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    if (!is_stubbed) {
        // Runs on the fence thread, the tick was submitted by Queue().
        scheduler.WaitSubmitted(wait_tick);
    }
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           NullQueryCache& query_cache_, Scheduler& scheduler_)
    : VideoCommon::FenceManager<FenceManagerParams>{rasterizer_, gpu_, texture_cache_,
                                                    buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock buffer_lock{buffer_cache.mutex};
    std::scoped_lock texture_lock{texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % VideoCore::Surface::BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

RasterizerMetal::RasterizerMetal(Core::Frontend::EmuWindow&, Tegra::GPU& gpu_,
                                 Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                 const Device& device_, Scheduler& scheduler_,
                                 StagingBufferPool& staging_pool_, BufferCache& buffer_cache_,
                                 TextureCacheRuntime& texture_cache_runtime_,
                                 TextureCache& texture_cache_, PipelineCache& pipeline_cache_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_}, scheduler{scheduler_},
      staging_pool{staging_pool_}, buffer_cache{buffer_cache_},
      texture_cache_runtime{texture_cache_runtime_}, texture_cache{texture_cache_},
      pipeline_cache{pipeline_cache_},
      shader_frontend{std::make_unique<ShaderFrontend>(device_memory_, pipeline_cache_)},
      accelerate_dma{buffer_cache_, texture_cache_},
      fence_manager{*this, gpu_, texture_cache_, buffer_cache_, query_cache, scheduler_} {
    // Apple GPUs from family 5 and every Mac2 GPU support 16 viewports.
    const DeviceFeatures& features = device.GetFeatures();
    max_viewports = features.apple_gpu_family >= 5 || features.apple_gpu_family == 0
                        ? Maxwell::NumViewports
                        : 1;
    @autoreleasepool {
        MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
        descriptor.supportArgumentBuffers = YES;
        fallback_sampler = [device.GetMTLDevice() newSamplerStateWithDescriptor:descriptor];
    }
}

RasterizerMetal::~RasterizerMetal() {
    scheduler.Finish();
}

void RasterizerMetal::CollectStageTextures(const Shader::Info& info, size_t stage,
                                           bool is_compute) {
    Tegra::MemoryManager& memory = *gpu_memory;
    bool via_header_index;
    std::function<GPUVAddr(u32)> cbuf_address;
    if (is_compute) {
        const auto& qmd = kepler_compute->launch_description;
        via_header_index = qmd.linked_tsc != 0;
        cbuf_address = [&qmd](u32 index) -> GPUVAddr {
            return index < qmd.const_buffer_config.size() ? qmd.const_buffer_config[index].Address()
                                                          : 0;
        };
    } else {
        via_header_index =
            maxwell3d->regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        cbuf_address = [&cbufs](u32 index) -> GPUVAddr {
            return index < cbufs.size() && cbufs[index].enabled ? cbufs[index].address : 0;
        };
    }
    const auto add_images = [&](const auto& descriptors, bool blacklist_written) {
        for (const auto& desc : descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle = ReadHandle(memory, desc, index, via_header_index, cbuf_address);
                bool blacklist = false;
                if constexpr (requires { desc.is_written; }) {
                    blacklist = blacklist_written && desc.is_written;
                }
                image_views.push_back({.index = handle.first, .blacklist = blacklist, .id = {}});
            }
        }
    };
    add_images(info.texture_buffer_descriptors, false);
    add_images(info.image_buffer_descriptors, false);
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const auto handle = ReadHandle(memory, desc, index, via_header_index, cbuf_address);
            image_views.push_back({.index = handle.first, .blacklist = false, .id = {}});
            samplers.push_back(texture_cache.GetSamplerId(handle.second, is_compute));
        }
    }
    add_images(info.image_descriptors, true);
}

id<MTLArgumentEncoder> RasterizerMetal::ArgumentEncoder(const ShaderModule& module) {
    const auto it = argument_encoders.find(&module);
    if (it != argument_encoders.end()) {
        return it->second;
    }
    id<MTLArgumentEncoder> encoder =
        [module.GetFunction() newArgumentEncoderWithBufferIndex:BufferSlot::DescriptorSet];
    if (!encoder) {
        LOG_ERROR(Render_Metal, "Failed to create the argument encoder of shader {:016x}",
                  module.Hash());
    }
    // Shader modules live as long as the pipeline cache, so the pointer key stays valid.
    argument_encoders.emplace(&module, encoder);
    return encoder;
}

bool RasterizerMetal::EncodeStageResources(const ShaderModule& module, const Shader::Info& info,
                                           const StageBindingSource& source, size_t& view_cursor,
                                           size_t& sampler_cursor, StageResources& out) {
    out.Reset();
    const TranslatedShader& translated = module.GetInfo();
    const BufferCacheRuntime& runtime = buffer_cache.runtime;

    id<MTLArgumentEncoder> encoder;
    if (translated.argument_buffer_length != 0) {
        encoder = ArgumentEncoder(module);
        if (!encoder) {
            return false;
        }
        const StagingBufferRef ref = staging_pool.RequestStream(
            encoder.encodedLength, (std::max)(static_cast<size_t>(encoder.alignment), size_t{16}));
        [encoder setArgumentBuffer:ref.buffer offset:ref.offset];
        out.argument_buffer = ref.buffer;
        out.argument_offset = ref.offset;
    }
    sizes_scratch.clear();
    if (encoder && translated.buffer_sizes_id) {
        sizes_scratch.assign(translated.argument_buffer_length, 0);
    }

    // Resources are sorted by binding, which follows the descriptor order of Shader::Info.
    const std::vector<ShaderResource>& resources = translated.resources;
    size_t next_resource = 0;
    bool layout_matches = true;
    const auto take = [&](ShaderResourceType type) -> const ShaderResource* {
        if (next_resource >= resources.size() || resources[next_resource].type != type) {
            layout_matches = false;
            return nullptr;
        }
        return &resources[next_resource++];
    };
    const auto set_buffer = [&](const BufferBinding& binding, u32 id, bool written) {
        if (!encoder || !binding.buffer) {
            return;
        }
        [encoder setBuffer:binding.buffer offset:binding.offset atIndex:id];
        (written ? out.writes : out.reads).push_back(binding.buffer);
        if (id < sizes_scratch.size()) {
            sizes_scratch[id] = static_cast<u32>(binding.size);
        }
    };
    const auto set_texture = [&](id<MTLTexture> texture, u32 id, bool written) {
        if (!encoder) {
            return;
        }
        if (!texture) {
            const ImageView& null_view = texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID);
            texture = null_view.Handle(Shader::TextureType::Color2D);
            written = false;
        }
        [encoder setTexture:texture atIndex:id];
        if (texture) {
            (written ? out.writes : out.reads).push_back(texture);
        }
    };

    u32 uniform_index = 0;
    for (const auto& desc : info.constant_buffer_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::UniformBuffer);
        for (u32 i = 0; i < desc.count; ++i, ++uniform_index) {
            const BufferBinding& binding = source.is_compute
                                               ? runtime.ComputeUniformBinding(uniform_index)
                                               : runtime.UniformBinding(source.stage, uniform_index);
            if (resource) {
                set_buffer(binding, resource->id + i, false);
            }
        }
    }
    u32 storage_index = 0;
    for (const auto& desc : info.storage_buffers_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::StorageBuffer);
        for (u32 i = 0; i < desc.count; ++i, ++storage_index) {
            const BufferBinding& binding = source.is_compute
                                               ? runtime.ComputeStorageBinding(storage_index)
                                               : runtime.StorageBinding(source.stage, storage_index);
            if (resource) {
                set_buffer(binding, resource->id + i, desc.is_written);
            }
        }
    }
    size_t texture_buffer_index = 0;
    const auto next_texture_buffer = [&]() -> id<MTLTexture> {
        ++view_cursor;
        return texture_buffer_index < source.texture_buffers.size()
                   ? source.texture_buffers[texture_buffer_index++]
                   : nil;
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::SampledImage);
        for (u32 i = 0; i < desc.count; ++i) {
            id<MTLTexture> texture = next_texture_buffer();
            if (resource) {
                set_texture(texture, resource->id + i, false);
            }
        }
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::StorageImage);
        for (u32 i = 0; i < desc.count; ++i) {
            id<MTLTexture> texture = next_texture_buffer();
            if (resource) {
                set_texture(texture, resource->id + i, desc.is_written);
            }
        }
    }
    for (const auto& desc : info.texture_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::CombinedImageSampler);
        for (u32 i = 0; i < desc.count; ++i) {
            const ImageView& view = texture_cache.GetImageView(image_views[view_cursor++].id);
            const Sampler* const sampler = &texture_cache.GetSampler(samplers[sampler_cursor++]);
            if (!resource || !encoder) {
                continue;
            }
            set_texture(view.Handle(desc.type), resource->id + i, false);
            id<MTLSamplerState> state = sampler && sampler->Handle() ? sampler->Handle()
                                                                     : fallback_sampler;
            [encoder setSamplerState:state atIndex:resource->secondary_id + i];
        }
    }
    for (const auto& desc : info.image_descriptors) {
        const ShaderResource* const resource = take(ShaderResourceType::StorageImage);
        for (u32 i = 0; i < desc.count; ++i) {
            const ImageView& view = texture_cache.GetImageView(image_views[view_cursor++].id);
            if (resource) {
                set_texture(view.Handle(desc.type), resource->id + i, desc.is_written);
            }
        }
    }
    if (!layout_matches || next_resource != resources.size()) {
        static bool logged = false;
        WarnOnce(logged, "Shader resource layout does not match the guest descriptors");
        return false;
    }
    if (encoder && translated.buffer_sizes_id) {
        const size_t bytes = sizes_scratch.size() * sizeof(u32);
        const StagingBufferRef ref = staging_pool.RequestStream(bytes, 16);
        std::memcpy(ref.mapped_span.data(), sizes_scratch.data(), bytes);
        [encoder setBuffer:ref.buffer offset:ref.offset atIndex:*translated.buffer_sizes_id];
        out.sizes_buffer = ref.buffer;
        out.sizes_offset = ref.offset;
    }
    return true;
}

void RasterizerMetal::PushConstantData(const Shader::Info& info, std::array<u8, 48>& out) const {
    out.fill(0);
    // Images are never rescaled on Metal: no rescaling bits and a down factor of one.
    const f32 down_factor = 1.0f;
    std::memcpy(out.data() + Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET,
                &down_factor, sizeof(down_factor));
    if (info.uses_render_area && maxwell3d) {
        const std::array<f32, 4> render_area{
            static_cast<f32>(maxwell3d->regs.surface_clip.width),
            static_cast<f32>(maxwell3d->regs.surface_clip.height), 0.0f, 0.0f};
        std::memcpy(out.data() + Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET,
                    render_area.data(), sizeof(render_area));
    }
}

void RasterizerMetal::BindRenderStage(id<MTLRenderCommandEncoder> encoder,
                                      const ShaderModule* module, const StageResources& resources,
                                      size_t slot, std::span<const u8> push_constants) {
    if (!module) {
        return;
    }
    const bool is_vertex = slot == 0;
    const TranslatedShader& info = module->GetInfo();
    // Draws of a pass mostly share these bindings, only changes reach the encoder.
    if (resources.argument_buffer &&
        (render_state.argument_buffers[slot] != resources.argument_buffer ||
         render_state.argument_offsets[slot] != resources.argument_offset)) {
        render_state.argument_buffers[slot] = resources.argument_buffer;
        render_state.argument_offsets[slot] = resources.argument_offset;
        if (is_vertex) {
            [encoder setVertexBuffer:resources.argument_buffer
                              offset:resources.argument_offset
                             atIndex:BufferSlot::DescriptorSet];
        } else {
            [encoder setFragmentBuffer:resources.argument_buffer
                                offset:resources.argument_offset
                               atIndex:BufferSlot::DescriptorSet];
        }
    }
    if (info.uses_push_constants) {
        std::array<u8, 48>& cached = render_state.push_constants[slot];
        const size_t size = (std::min)(cached.size(), push_constants.size());
        if (!render_state.push_constants_valid[slot] ||
            std::memcmp(cached.data(), push_constants.data(), size) != 0) {
            std::copy_n(push_constants.begin(), size, cached.begin());
            render_state.push_constants_valid[slot] = true;
            if (is_vertex) {
                [encoder setVertexBytes:push_constants.data()
                                 length:push_constants.size()
                                atIndex:BufferSlot::PushConstants];
            } else {
                [encoder setFragmentBytes:push_constants.data()
                                   length:push_constants.size()
                                  atIndex:BufferSlot::PushConstants];
            }
        }
    }
    if (info.needs_buffer_size_buffer && resources.sizes_buffer &&
        (render_state.sizes_buffers[slot] != resources.sizes_buffer ||
         render_state.sizes_offsets[slot] != resources.sizes_offset)) {
        render_state.sizes_buffers[slot] = resources.sizes_buffer;
        render_state.sizes_offsets[slot] = resources.sizes_offset;
        if (is_vertex) {
            [encoder setVertexBuffer:resources.sizes_buffer
                              offset:resources.sizes_offset
                             atIndex:BufferSlot::BufferSizes];
        } else {
            [encoder setFragmentBuffer:resources.sizes_buffer
                                offset:resources.sizes_offset
                               atIndex:BufferSlot::BufferSizes];
        }
    }
    // Resources only referenced through argument buffers must be made resident explicitly.
    // Residency lasts for the whole pass, so lists already declared are not repeated: the
    // driver walks every list it is given.
    const MTLRenderStages stages = is_vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
    if (!resources.reads.empty()) {
        const u64 hash = HashResourceList(resources.reads);
        if (hash != render_state.resident_reads[slot]) {
            render_state.resident_reads[slot] = hash;
            [encoder useResources:resources.reads.data()
                            count:resources.reads.size()
                            usage:MTLResourceUsageRead
                           stages:stages];
        }
    }
    if (!resources.writes.empty()) {
        const u64 hash = HashResourceList(resources.writes);
        if (hash != render_state.resident_writes[slot]) {
            render_state.resident_writes[slot] = hash;
            [encoder useResources:resources.writes.data()
                            count:resources.writes.size()
                            usage:MTLResourceUsageRead | MTLResourceUsageWrite
                           stages:stages];
        }
    }
}

GraphicsPipelineKey RasterizerMetal::MakeGraphicsPipelineKey(const GraphicsProgram& program,
                                                             const Framebuffer& framebuffer) const {
    const auto& regs = maxwell3d->regs;
    GraphicsPipelineKey key{};
    key.vertex_shader = program.vertex_hash;
    key.fragment_shader = program.fragment_hash;

    const Shader::Info& vertex_info = program.infos[VERTEX_STAGE];
    std::array<bool, BufferSlot::NumVertexBuffers> used_buffers{};
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& attribute = regs.vertex_attrib_format[index];
        VertexAttributeKey& out = key.attributes[index];
        out.format = MTLVertexFormatInvalid;
        if (attribute.constant || attribute.size == Maxwell::VertexAttribute::Size::Invalid ||
            !vertex_info.loads.Generic(index) || attribute.buffer >= BufferSlot::NumVertexBuffers) {
            continue;
        }
        out.format = static_cast<u16>(VertexFormat(attribute.type, attribute.size));
        out.offset = static_cast<u16>(attribute.offset);
        out.buffer = static_cast<u16>(attribute.buffer);
        used_buffers[attribute.buffer] = out.format != MTLVertexFormatInvalid;
    }
    for (size_t index = 0; index < BufferSlot::NumVertexBuffers; ++index) {
        if (used_buffers[index]) {
            key.vertex_buffers[index] = MakeVertexBufferKey(regs, index);
        }
    }

    const auto& formats = framebuffer.ColorFormats();
    for (size_t index = 0; index < NumColorAttachments; ++index) {
        ColorAttachmentKey& out = key.color_attachments[index];
        out.pixel_format = static_cast<u16>(formats[index]);
        if (formats[index] == MTLPixelFormatInvalid) {
            continue;
        }
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        out.write_mask = static_cast<u8>((mask.R ? MTLColorWriteMaskRed : 0) |
                                         (mask.G ? MTLColorWriteMaskGreen : 0) |
                                         (mask.B ? MTLColorWriteMaskBlue : 0) |
                                         (mask.A ? MTLColorWriteMaskAlpha : 0));
        bool is_integer = false;
        if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
            is_integer = VideoCore::Surface::IsPixelFormatInteger(
                VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format));
        }
        if (!regs.blend.enable[index] || is_integer) {
            out.blend_enable = 0;
            out.rgb_blend_op = MTLBlendOperationAdd;
            out.alpha_blend_op = MTLBlendOperationAdd;
            out.src_rgb_factor = MTLBlendFactorOne;
            out.dst_rgb_factor = MTLBlendFactorZero;
            out.src_alpha_factor = MTLBlendFactorOne;
            out.dst_alpha_factor = MTLBlendFactorZero;
            continue;
        }
        const auto setup = [&](const auto& blend) {
            out.blend_enable = 1;
            out.rgb_blend_op = static_cast<u8>(BlendOperation(blend.color_op));
            out.alpha_blend_op = static_cast<u8>(BlendOperation(blend.alpha_op));
            out.src_rgb_factor = static_cast<u8>(BlendFactor(blend.color_source));
            out.dst_rgb_factor = static_cast<u8>(BlendFactor(blend.color_dest));
            out.src_alpha_factor = static_cast<u8>(BlendFactor(blend.alpha_source));
            out.dst_alpha_factor = static_cast<u8>(BlendFactor(blend.alpha_dest));
        };
        if (regs.blend_per_target_enabled) {
            setup(regs.blend_per_target[index]);
        } else {
            setup(regs.blend);
        }
    }
    const MTLPixelFormat depth_format = framebuffer.DepthFormat();
    key.depth_format = static_cast<u16>(HasDepthAspect(depth_format) ? depth_format
                                                                     : MTLPixelFormatInvalid);
    key.stencil_format =
        static_cast<u16>(framebuffer.HasStencil() ? depth_format : MTLPixelFormatInvalid);
    key.sample_count = static_cast<u8>((std::max)(framebuffer.Samples(), 1U));
    // Only layered rendering requires a topology class, leaving it unspecified otherwise keeps
    // one pipeline for every primitive type.
    key.topology_class = static_cast<u8>(
        program.layered ? TopologyClass(maxwell3d->draw_manager.draw_state.topology)
                        : MTLPrimitiveTopologyClassUnspecified);
    key.alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1 : 0;
    key.alpha_to_one = regs.anti_alias_alpha_control.alpha_to_one != 0 ? 1 : 0;
    key.rasterization_enabled = regs.rasterize_enable != 0 ? 1 : 0;
    return key;
}

bool RasterizerMetal::VertexStreamsMatch(const GraphicsPipelineKey& key) const {
    const auto& regs = maxwell3d->regs;
    std::array<bool, BufferSlot::NumVertexBuffers> used{};
    for (const VertexAttributeKey& attribute : key.attributes) {
        if (attribute.format != MTLVertexFormatInvalid && attribute.buffer < used.size()) {
            used[attribute.buffer] = true;
        }
    }
    for (size_t index = 0; index < used.size(); ++index) {
        if (!used[index]) {
            continue;
        }
        const VertexBufferKey current = MakeVertexBufferKey(regs, index);
        if (std::memcmp(&current, &key.vertex_buffers[index], sizeof(current)) != 0) {
            return false;
        }
    }
    return true;
}

id<MTLDepthStencilState> RasterizerMetal::DepthStencilState(bool has_depth, bool has_stencil) {
    const auto& regs = maxwell3d->regs;
    const bool depth_test = has_depth && regs.depth_test_enable != 0;
    const bool depth_write = has_depth && regs.depth_test_enable != 0 &&
                             regs.depth_write_enabled != 0;
    const bool stencil = has_stencil && regs.stencil_enable != 0;
    const bool two_side = regs.stencil_two_side_enable != 0;
    const auto& back_op = two_side ? regs.stencil_back_op : regs.stencil_front_op;
    const u32 back_func_mask = two_side ? regs.stencil_back_func_mask : regs.stencil_front_func_mask;
    const u32 back_mask = two_side ? regs.stencil_back_mask : regs.stencil_front_mask;

    u64 key = 0;
    u32 shift = 0;
    const auto pack = [&](u64 value, u32 bits) {
        key |= (value & ((u64{1} << bits) - 1)) << shift;
        shift += bits;
    };
    pack(depth_test, 1);
    pack(depth_write, 1);
    pack(depth_test ? PackComparisonOp(regs.depth_test_func)
                    : static_cast<u32>(MTLCompareFunctionAlways),
         3);
    pack(stencil, 1);
    if (stencil) {
        pack(PackStencilOp(regs.stencil_front_op.fail), 3);
        pack(PackStencilOp(regs.stencil_front_op.zfail), 3);
        pack(PackStencilOp(regs.stencil_front_op.zpass), 3);
        pack(PackComparisonOp(regs.stencil_front_op.func), 3);
        pack(PackStencilOp(back_op.fail), 3);
        pack(PackStencilOp(back_op.zfail), 3);
        pack(PackStencilOp(back_op.zpass), 3);
        pack(PackComparisonOp(back_op.func), 3);
        pack(regs.stencil_front_func_mask, 8);
        pack(regs.stencil_front_mask, 8);
        pack(back_func_mask, 8);
        pack(back_mask, 8);
    }
    const auto it = depth_stencil_states.find(key);
    if (it != depth_stencil_states.end()) {
        return it->second;
    }
    id<MTLDepthStencilState> state;
    @autoreleasepool {
        MTLDepthStencilDescriptor* descriptor = [[MTLDepthStencilDescriptor alloc] init];
        descriptor.depthCompareFunction =
            depth_test ? CompareFunction(regs.depth_test_func) : MTLCompareFunctionAlways;
        descriptor.depthWriteEnabled = depth_write;
        if (stencil) {
            const auto make_face = [](const Maxwell::StencilOp& op, u32 read_mask,
                                      u32 write_mask) {
                MTLStencilDescriptor* face = [[MTLStencilDescriptor alloc] init];
                face.stencilCompareFunction = CompareFunction(op.func);
                face.stencilFailureOperation = static_cast<MTLStencilOperation>(PackStencilOp(op.fail));
                face.depthFailureOperation = static_cast<MTLStencilOperation>(PackStencilOp(op.zfail));
                face.depthStencilPassOperation =
                    static_cast<MTLStencilOperation>(PackStencilOp(op.zpass));
                face.readMask = read_mask & 0xFF;
                face.writeMask = write_mask & 0xFF;
                return face;
            };
            descriptor.frontFaceStencil = make_face(regs.stencil_front_op,
                                                    regs.stencil_front_func_mask,
                                                    regs.stencil_front_mask);
            descriptor.backFaceStencil = make_face(back_op, back_func_mask, back_mask);
        }
        state = [device.GetMTLDevice() newDepthStencilStateWithDescriptor:descriptor];
    }
    depth_stencil_states.emplace(key, state);
    return state;
}

bool RasterizerMetal::UpdateDynamicStates(id<MTLRenderCommandEncoder> encoder,
                                          const Framebuffer& framebuffer) {
    const auto& regs = maxwell3d->regs;
    auto& dirty = maxwell3d->dirty.flags;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const u32 fb_width = framebuffer.Width();
    const u32 fb_height = framebuffer.Height();
    const bool has_depth = HasDepthAspect(framebuffer.DepthFormat());
    const bool has_stencil = framebuffer.HasStencil();

    // Decoding the registers is the expensive part, it only happens when they changed.
    const bool recompute = !derived.dynamic_valid || dirty[Dirty::Viewports] ||
                           dirty[Dirty::Scissors] || dirty[Dirty::RasterState] ||
                           dirty[Dirty::DepthStencil] || derived.fb_width != fb_width ||
                           derived.fb_height != fb_height || derived.has_depth != has_depth ||
                           derived.has_stencil != has_stencil ||
                           derived.dynamic_topology != topology;
    if (recompute) {
        dirty[Dirty::Viewports] = false;
        dirty[Dirty::Scissors] = false;
        dirty[Dirty::RasterState] = false;
        dirty[Dirty::DepthStencil] = false;
        derived.dynamic_valid = true;
        derived.fb_width = fb_width;
        derived.fb_height = fb_height;
        derived.has_depth = has_depth;
        derived.has_stencil = has_stencil;
        derived.dynamic_topology = topology;
        derived.skip_draw = false;
        RenderEncoderState::Dynamic& computed = derived.dynamic;

        const size_t count = max_viewports;
        if (!regs.viewport_scale_offset_enabled) {
            f32 y = static_cast<f32>(regs.surface_clip.y);
            f32 height = (std::max)(1.0f, static_cast<f32>(regs.surface_clip.height));
            if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
                y += height;
                height = -height;
            }
            const MTLViewport viewport{
                .originX = static_cast<f64>(regs.surface_clip.x),
                .originY = y,
                .width = (std::max)(1.0, static_cast<f64>(regs.surface_clip.width)),
                .height = height,
                .znear = 0.0,
                .zfar = 1.0,
            };
            s64 scissor_y = regs.surface_clip.y;
            const s64 scissor_height = (std::max)(1U, static_cast<u32>(regs.surface_clip.height));
            if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
                scissor_y = s64{regs.surface_clip.height} - (scissor_y + scissor_height);
            }
            const MTLScissorRect scissor = ClampScissor(
                regs.surface_clip.x, scissor_y,
                (std::max)(1U, static_cast<u32>(regs.surface_clip.width)), scissor_height,
                fb_width, fb_height);
            computed.viewports.fill(viewport);
            computed.scissors.fill(scissor);
        } else {
            for (size_t index = 0; index < count; ++index) {
                computed.viewports[index] = GetViewport(regs, index);
                computed.scissors[index] = GetScissor(regs, index, fb_width, fb_height);
            }
        }
        computed.viewport_count = count;
        if (computed.scissors[0].width == 0 || computed.scissors[0].height == 0) {
            derived.skip_draw = true;
        }

        const bool is_triangles = TopologyClass(topology) == MTLPrimitiveTopologyClassTriangle;
        computed.cull_mode = MTLCullModeNone;
        if (regs.gl_cull_test_enabled) {
            switch (regs.gl_cull_face) {
            case Maxwell::CullFace::Front:
                computed.cull_mode = MTLCullModeFront;
                break;
            case Maxwell::CullFace::Back:
                computed.cull_mode = MTLCullModeBack;
                break;
            case Maxwell::CullFace::FrontAndBack:
                // Metal can't cull both faces, every triangle would be discarded.
                if (is_triangles) {
                    derived.skip_draw = true;
                }
                break;
            }
        }
        bool clockwise = regs.gl_front_face == Maxwell::FrontFace::ClockWise;
        if (regs.window_origin.flip_y != 0) {
            clockwise = !clockwise;
        }
        computed.winding = clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise;
        computed.fill_mode = regs.polygon_mode_front == Maxwell::PolygonMode::Line
                                 ? MTLTriangleFillModeLines
                                 : MTLTriangleFillModeFill;

        using GeometryClip = Maxwell::ViewportClipControl::GeometryClip;
        const GeometryClip clip = regs.viewport_clip_control.geometry_clip;
        const bool depth_clamp = !(clip == GeometryClip::Passthrough ||
                                   clip == GeometryClip::FrustumXYZ || clip == GeometryClip::FrustumZ);
        computed.clip_mode = depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;

        bool bias_enable;
        switch (TopologyClass(topology)) {
        case MTLPrimitiveTopologyClassPoint:
            bias_enable = regs.polygon_offset_point_enable != 0;
            break;
        case MTLPrimitiveTopologyClassLine:
            bias_enable = regs.polygon_offset_line_enable != 0;
            break;
        default:
            bias_enable = regs.polygon_offset_fill_enable != 0;
            break;
        }
        computed.depth_bias = bias_enable ? std::array<f32, 3>{regs.depth_bias / 2.0f,
                                                               regs.slope_scale_depth_bias,
                                                               regs.depth_bias_clamp}
                                          : std::array<f32, 3>{};
        computed.blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                regs.blend_color.a};
        const u32 front_ref = regs.stencil_front_ref & 0xFF;
        const u32 back_ref = (regs.stencil_two_side_enable ? regs.stencil_back_ref
                                                           : regs.stencil_front_ref) &
                             0xFF;
        computed.stencil_refs = front_ref | (back_ref << 8);
        derived.depth_stencil = DepthStencilState(has_depth, has_stencil);

        if (regs.depth_bounds_enable) {
            static bool logged = false;
            WarnOnce(logged, "Depth bounds test is not supported by Metal");
        }
    }
    if (derived.skip_draw) {
        return false;
    }

    // Draws in the same pass usually share most of this state, only changes are encoded.
    const RenderEncoderState::Dynamic& computed = derived.dynamic;
    RenderEncoderState::Dynamic& cached = render_state.dynamic;
    const bool valid = render_state.dynamic_valid;
    const auto changed = [valid](auto& current, const auto& value) {
        if (valid && std::memcmp(&current, &value, sizeof(value)) == 0) {
            return false;
        }
        current = value;
        return true;
    };
    const size_t count = computed.viewport_count;
    if (!valid || cached.viewport_count != count ||
        std::memcmp(cached.viewports.data(), computed.viewports.data(),
                    count * sizeof(MTLViewport)) != 0) {
        std::copy_n(computed.viewports.begin(), count, cached.viewports.begin());
        [encoder setViewports:computed.viewports.data() count:count];
    }
    if (!valid || cached.viewport_count != count ||
        std::memcmp(cached.scissors.data(), computed.scissors.data(),
                    count * sizeof(MTLScissorRect)) != 0) {
        std::copy_n(computed.scissors.begin(), count, cached.scissors.begin());
        [encoder setScissorRects:computed.scissors.data() count:count];
    }
    cached.viewport_count = count;
    if (changed(cached.cull_mode, computed.cull_mode)) {
        [encoder setCullMode:computed.cull_mode];
    }
    if (changed(cached.winding, computed.winding)) {
        [encoder setFrontFacingWinding:computed.winding];
    }
    if (changed(cached.fill_mode, computed.fill_mode)) {
        [encoder setTriangleFillMode:computed.fill_mode];
    }
    if (changed(cached.clip_mode, computed.clip_mode)) {
        [encoder setDepthClipMode:computed.clip_mode];
    }
    if (changed(cached.depth_bias, computed.depth_bias)) {
        [encoder setDepthBias:computed.depth_bias[0]
                   slopeScale:computed.depth_bias[1]
                        clamp:computed.depth_bias[2]];
    }
    if (changed(cached.blend_color, computed.blend_color)) {
        [encoder setBlendColorRed:computed.blend_color[0]
                            green:computed.blend_color[1]
                             blue:computed.blend_color[2]
                            alpha:computed.blend_color[3]];
    }
    if (derived.depth_stencil != render_state.depth_stencil) {
        [encoder setDepthStencilState:derived.depth_stencil];
        render_state.depth_stencil = derived.depth_stencil;
    }
    if (changed(cached.stencil_refs, computed.stencil_refs)) {
        [encoder setStencilFrontReferenceValue:computed.stencil_refs & 0xFF
                            backReferenceValue:(computed.stencil_refs >> 8) & 0xFF];
    }
    render_state.dynamic_valid = true;
    return true;
}

bool RasterizerMetal::PrepareDraw(bool is_indexed, DrawSetup& setup) {
    const GraphicsProgram& program = *current_program;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    auto& dirty = maxwell3d->dirty.flags;
    // Vertex buffer addresses change for most draws; only strides and instancing are part of
    // the pipeline key, checked against the registers when the buffer cache consumes the flag.
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    bool draw_writes_memory = false;
    if (!PrimitiveType(draw_state.topology)) {
        static bool logged = false;
        WarnOnce(logged, "Adjacency and patch topologies are not supported by Metal");
        return false;
    }

    // Constant buffers of a folded geometry shader are read by the vertex stage from slots the
    // vertex program doesn't use. The guest vertex bindings of those slots are restored after
    // the draw, Maxwell3D only rebinds them when the guest writes the registers again.
    const auto& shader_stages = maxwell3d->state.shader_stages;
    const auto bind_vertex_cbuf = [&](u32 slot, const Tegra::Engines::ConstBufferInfo& cbuf) {
        if (cbuf.enabled && gpu_memory->GpuToCpuAddress(cbuf.address)) {
            buffer_cache.BindGraphicsUniformBuffer(VERTEX_STAGE, slot, cbuf.address, cbuf.size);
        } else {
            buffer_cache.DisableGraphicsUniformBuffer(VERTEX_STAGE, slot);
        }
    };
    for (const auto& [geometry_slot, vertex_slot] : program.geometry_cbuf_remap) {
        bind_vertex_cbuf(vertex_slot,
                         shader_stages[GEOMETRY_STAGE].const_buffers[geometry_slot]);
    }
    SCOPE_EXIT {
        if (program.geometry_cbuf_remap.empty()) {
            return;
        }
        // The locks may have been released for the encoder work, the buffer cache is only
        // touched with them held.
        if (setup.buffer_lock && !setup.buffer_lock->owns_lock()) {
            setup.buffer_lock->lock();
        }
        for (const auto& remap : program.geometry_cbuf_remap) {
            const u32 slot = remap.second;
            bind_vertex_cbuf(slot, shader_stages[VERTEX_STAGE].const_buffers[slot]);
        }
    };

    image_views.clear();
    samplers.clear();
    texture_cache.SynchronizeDescriptors(false);
    buffer_cache.SetUniformBuffersState(program.enabled_uniform_buffer_masks,
                                        &program.uniform_buffer_sizes);
    for (const size_t stage : RENDER_STAGES) {
        if (!program.enabled[stage]) {
            continue;
        }
        const Shader::Info& info = program.infos[stage];
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        size_t ssbo_index = 0;
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written);
            draw_writes_memory |= desc.is_written;
            ++ssbo_index;
        }
        for (const auto& desc : info.image_descriptors) {
            draw_writes_memory |= desc.is_written;
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            draw_writes_memory |= desc.is_written;
        }
        CollectStageTextures(info, stage, false);
    }
    texture_cache.FillImageViews(std::span(image_views.data(), image_views.size()), false, true);
    FlushPendingClearsForBoundViews();

    size_t view_index = 0;
    for (const size_t stage : RENDER_STAGES) {
        if (!program.enabled[stage]) {
            continue;
        }
        const Shader::Info& info = program.infos[stage];
        buffer_cache.UnbindGraphicsTextureBuffers(stage);
        size_t index = 0;
        const auto add_buffers = [&](const auto& descriptors, bool is_image) {
            for (const auto& desc : descriptors) {
                for (u32 i = 0; i < desc.count; ++i) {
                    ImageView& view = texture_cache.GetImageView(image_views[view_index++].id);
                    bool is_written = false;
                    if constexpr (requires { desc.is_written; }) {
                        is_written = desc.is_written;
                    }
                    buffer_cache.BindGraphicsTextureBuffer(stage, index++, view.GpuAddr(),
                                                           view.BufferSize(), view.format,
                                                           is_written, is_image);
                }
            }
        };
        add_buffers(info.texture_buffer_descriptors, false);
        add_buffers(info.image_buffer_descriptors, true);
        view_index += Shader::NumDescriptors(info.texture_descriptors);
        view_index += Shader::NumDescriptors(info.image_descriptors);
    }

    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);
    BufferCacheRuntime& runtime = buffer_cache.runtime;
    for (const size_t stage : RENDER_STAGES) {
        stage_texture_buffers[stage].clear();
        if (!program.enabled[stage]) {
            continue;
        }
        runtime.ResetTextureBuffers();
        buffer_cache.BindHostStageBuffers(stage);
        const auto bound = runtime.TextureBuffers();
        stage_texture_buffers[stage].assign(bound.begin(), bound.end());
    }
    if (buffer_cache.any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    // The buffer cache is done with this draw; the texture cache still has work below.
    if (setup.buffer_lock) {
        setup.buffer_lock->unlock();
    }
    texture_cache.UpdateRenderTargets(false);
    texture_cache.CheckFeedbackLoop(
        std::span<const ImageViewInOut>{image_views.data(), image_views.size()});

    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (!framebuffer || framebuffer->Width() == 0 || framebuffer->Height() == 0) {
        return false;
    }
    ResolveFeedbackLoop(*framebuffer);
    // Building and hashing the pipeline key decodes dozens of registers, consecutive draws
    // nearly always reuse the previous pipeline.
    const FramebufferFormats fb_formats{
        .color = framebuffer->ColorFormats(),
        .depth = framebuffer->DepthFormat(),
        .samples = framebuffer->Samples(),
        .has_stencil = framebuffer->HasStencil(),
    };
    GraphicsPipeline* pipeline = derived.pipeline;
    bool reuse_pipeline =
        pipeline && !dirty[Dirty::PipelineState] && derived.program == &program &&
        derived.vertex_hash == program.vertex_hash &&
        derived.fragment_hash == program.fragment_hash && derived.framebuffer == fb_formats &&
        (!program.layered || derived.pipeline_topology == draw_state.topology);
    if (reuse_pipeline && vertex_buffers_dirty) {
        reuse_pipeline = VertexStreamsMatch(derived.key);
    }
    if (!reuse_pipeline) {
        derived.key = MakeGraphicsPipelineKey(program, *framebuffer);
        pipeline = pipeline_cache.GetGraphicsPipeline(derived.key);
        ++stats.pipeline_lookups;
        dirty[Dirty::PipelineState] = false;
        derived.pipeline = pipeline;
        derived.program = &program;
        derived.vertex_hash = program.vertex_hash;
        derived.fragment_hash = program.fragment_hash;
        derived.framebuffer = fb_formats;
        derived.pipeline_topology = draw_state.topology;
    }
    if (!pipeline) {
        return false;
    }
    if (pipeline->Status() == PipelineStatus::Building) {
        // Same heuristic as Vulkan: skip draws while shaders build asynchronously, except
        // small full screen passes that are likely used once to build textures.
        const bool small_draw =
            draw_state.index_buffer.count <= 6 || draw_state.vertex_buffer.count <= 6;
        if (Settings::values.use_asynchronous_shaders.GetValue() &&
            (maxwell3d->regs.zeta_enable || !small_draw)) {
            return false;
        }
    }
    if (!pipeline->WaitUntilBuilt()) {
        return false;
    }

    // Argument buffers are written before the encoder is requested: the caches may still record
    // blits until this point, which would end an open render pass.
    size_t view_cursor = 0;
    size_t sampler_cursor = 0;
    const std::array<const ShaderModule*, 2> modules{pipeline->VertexShader(),
                                                     pipeline->FragmentShader()};
    for (size_t i = 0; i < RENDER_STAGES.size(); ++i) {
        const size_t stage = RENDER_STAGES[i];
        if (!program.enabled[stage]) {
            continue;
        }
        const StageBindingSource source{
            .stage = stage,
            .is_compute = false,
            .texture_buffers = stage_texture_buffers[stage],
        };
        if (!modules[i]) {
            // Fragment shaders are dropped when rasterization is disabled.
            const Shader::Info& info = program.infos[stage];
            view_cursor += Shader::NumDescriptors(info.texture_buffer_descriptors) +
                           Shader::NumDescriptors(info.image_buffer_descriptors) +
                           Shader::NumDescriptors(info.texture_descriptors) +
                           Shader::NumDescriptors(info.image_descriptors);
            sampler_cursor += Shader::NumDescriptors(info.texture_descriptors);
            continue;
        }
        if (!EncodeStageResources(*modules[i], program.infos[stage], source, view_cursor,
                                  sampler_cursor, render_resources[i])) {
            return false;
        }
    }

    // The texture cache is done as well: what follows touches the scheduler and the runtime
    // only, while guest threads block on these locks for every write to cached memory.
    if (setup.texture_lock) {
        setup.texture_lock->unlock();
    }
    id<MTLRenderCommandEncoder> encoder = scheduler.RenderEncoder(framebuffer->Descriptor());
    if (!encoder) {
        return false;
    }
    if (render_state.generation != scheduler.EncoderGeneration()) {
        render_state = RenderEncoderState{.generation = scheduler.EncoderGeneration()};
    }
    if (draw_writes_memory) {
        // Barriers only need to end this pass when shaders in it wrote memory.
        pass_writes_generation = scheduler.EncoderGeneration();
    }
    if (render_state.pipeline != pipeline->GetState()) {
        [encoder setRenderPipelineState:pipeline->GetState()];
        render_state.pipeline = pipeline->GetState();
    }
    if (!UpdateDynamicStates(encoder, *framebuffer)) {
        return false;
    }
    {
        // Sampling the depth buffer of the pass only needs the pass to end after writes to the
        // sampled aspect, tracked separately for depth and stencil.
        const auto& regs = maxwell3d->regs;
        const bool writes_depth = derived.has_depth && regs.depth_test_enable != 0 &&
                                  regs.depth_write_enabled != 0;
        const u32 stencil_mask = regs.stencil_two_side_enable != 0
                                     ? (regs.stencil_front_mask | regs.stencil_back_mask)
                                     : regs.stencil_front_mask;
        const bool writes_stencil =
            derived.has_stencil && regs.stencil_enable != 0 && (stencil_mask & 0xFF) != 0;
        if (writes_depth) {
            scheduler.MarkPassDepthWrite();
        }
        if (writes_stencil) {
            scheduler.MarkPassStencilWrite();
        }
    }
    runtime.ApplyVertexBuffers(encoder, BufferSlot::FirstVertexBuffer,
                               render_state.vertex_buffers);

    for (size_t i = 0; i < RENDER_STAGES.size(); ++i) {
        const size_t stage = RENDER_STAGES[i];
        if (!program.enabled[stage] || !modules[i]) {
            continue;
        }
        std::array<u8, 48> push_constants;
        PushConstantData(program.infos[stage], push_constants);
        BindRenderStage(encoder, modules[i], render_resources[i], i, push_constants);
    }
    current_pipeline = pipeline;
    setup.encoder = encoder;
    setup.framebuffer = framebuffer;
    return true;
}

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    @autoreleasepool {
        SCOPE_EXIT {
            gpu.TickWork();
        };
        FlushWork();
        // Locked in a fixed order rather than with std::lock, whose try-lock retries yield to the
        // guest threads contending these mutexes. Pending invalidations are applied under the
        // locks, so guest threads get no window between them and the draw. PrepareDraw
        // releases both once the caches are done, before the encoder work.
        std::unique_lock buffer_lock{buffer_cache.mutex};
        std::unique_lock texture_lock{texture_cache.mutex};
        gpu_memory->FlushCaching();

        current_program = shader_frontend->CurrentGraphicsProgram();
        if (!current_program || instance_count == 0) {
            return;
        }
        ++stats.draws;
        DrawSetup setup;
        setup.buffer_lock = &buffer_lock;
        setup.texture_lock = &texture_lock;
        if (!PrepareDraw(is_indexed, setup)) {
            return;
        }
        id<MTLRenderCommandEncoder> encoder = setup.encoder;
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const auto topology = draw_state.topology;
        const MTLPrimitiveType primitive_type = *PrimitiveType(topology);
        const NSUInteger base_instance = draw_state.base_instance;

        const auto draw_indices = [&](std::span<const u32> indices, NSInteger base_vertex) {
            if (indices.empty()) {
                return;
            }
            const StagingBufferRef ref = staging_pool.RequestStream(indices.size_bytes(), 4);
            std::memcpy(ref.mapped_span.data(), indices.data(), indices.size_bytes());
            [encoder drawIndexedPrimitives:primitive_type
                                indexCount:indices.size()
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:ref.buffer
                         indexBufferOffset:ref.offset
                             instanceCount:instance_count
                                baseVertex:base_vertex
                              baseInstance:base_instance];
        };

        if (is_indexed) {
            const IndexBufferBinding& binding = buffer_cache.runtime.IndexBinding();
            if (!binding.buffer || binding.count == 0) {
                return;
            }
            const NSInteger base_vertex = static_cast<s32>(draw_state.base_index);
            if (!binding.converted_quads && NeedsIndexConversion(topology)) {
                // Shared memory: the guest indices are read in place.
                const u8* const data = static_cast<const u8*>(binding.buffer.contents) + binding.offset;
                const bool is_u16 = binding.type == MTLIndexTypeUInt16;
                GenerateIndices(
                    topology, binding.count,
                    [&](u32 i) -> u32 {
                        if (is_u16) {
                            u16 value;
                            std::memcpy(&value, data + size_t{i} * 2, sizeof(value));
                            return value;
                        }
                        u32 value;
                        std::memcpy(&value, data + size_t{i} * 4, sizeof(value));
                        return value;
                    },
                    index_scratch);
                draw_indices(index_scratch, base_vertex);
                return;
            }
            // Converted quads hold absolute vertex numbers.
            [encoder drawIndexedPrimitives:primitive_type
                                indexCount:binding.count
                                 indexType:binding.type
                               indexBuffer:binding.buffer
                         indexBufferOffset:binding.offset
                             instanceCount:instance_count
                                baseVertex:binding.converted_quads ? 0 : base_vertex
                              baseInstance:base_instance];
            return;
        }
        const u32 first = draw_state.vertex_buffer.first;
        const u32 count = draw_state.vertex_buffer.count;
        if (count == 0) {
            return;
        }
        if (NeedsIndexConversion(topology)) {
            GenerateIndices(topology, count, [first](u32 i) { return first + i; }, index_scratch);
            draw_indices(index_scratch, 0);
            return;
        }
        [encoder drawPrimitives:primitive_type
                    vertexStart:first
                    vertexCount:count
                  instanceCount:instance_count
                   baseInstance:base_instance];
    }
}

void RasterizerMetal::DrawIndirect() {
    @autoreleasepool {
        SCOPE_EXIT {
            gpu.TickWork();
        };
        FlushWork();
        std::scoped_lock buffer_lock{buffer_cache.mutex};
        std::scoped_lock texture_lock{texture_cache.mutex};
        gpu_memory->FlushCaching();

        const auto& params = maxwell3d->draw_manager.indirect_state;
        current_program = shader_frontend->CurrentGraphicsProgram();
        if (!current_program) {
            return;
        }
        if (params.is_byte_count || NeedsIndexConversion(maxwell3d->draw_manager.draw_state.topology)) {
            static bool logged = false;
            WarnOnce(logged, "Byte count indirect draws and indirect draws of converted topologies "
                             "are not implemented");
            return;
        }
        ++stats.draws;
        buffer_cache.SetDrawIndirect(&params);
        SCOPE_EXIT {
            buffer_cache.SetDrawIndirect(nullptr);
        };
        DrawSetup setup;
        if (!PrepareDraw(params.is_indexed, setup)) {
            return;
        }
        const auto [indirect_buffer, indirect_offset] = buffer_cache.GetDrawIndirectBuffer();
        if (!indirect_buffer || !indirect_buffer->Handle()) {
            return;
        }
        size_t draw_count = params.max_draw_counts;
        if (params.include_count) {
            // TODO: Use the GPU count once indirect command buffers are implemented.
            draw_count = (std::min<size_t>)(draw_count,
                                            gpu_memory->Read<u32>(params.count_start_address));
        }
        const MTLPrimitiveType primitive_type =
            *PrimitiveType(maxwell3d->draw_manager.draw_state.topology);
        const IndexBufferBinding& binding = buffer_cache.runtime.IndexBinding();
        for (size_t draw = 0; draw < draw_count; ++draw) {
            const NSUInteger offset = indirect_offset + draw * params.stride;
            if (params.is_indexed) {
                if (!binding.buffer) {
                    return;
                }
                // MTLDrawIndexedPrimitivesIndirectArguments matches VkDrawIndexedIndirectCommand.
                [setup.encoder drawIndexedPrimitives:primitive_type
                                           indexType:binding.type
                                         indexBuffer:binding.buffer
                                   indexBufferOffset:binding.offset
                                      indirectBuffer:indirect_buffer->Handle()
                                indirectBufferOffset:offset];
            } else {
                [setup.encoder drawPrimitives:primitive_type
                               indirectBuffer:indirect_buffer->Handle()
                         indirectBufferOffset:offset];
            }
        }
    }
}

id<MTLFunction> RasterizerMetal::UtilityFunction(const std::string& name, u32 attachment,
                                                 u32 kind) {
    const u64 key = (u64{kind} << 32) | (u64{attachment} << 24) |
                    (Common::CityHash64(name.data(), name.size()) & 0xFFFFFF);
    const auto it = utility_functions.find(key);
    if (it != utility_functions.end()) {
        return it->second;
    }
    id<MTLFunction> function;
    @autoreleasepool {
        std::string error;
        id<MTLLibrary> library;
        if (kind <= UTIL_KIND_SINT) {
            static constexpr std::array types{"float4", "uint4", "int4"};
            static constexpr std::array values{"p.color", "p.ucolor", "as_type<int4>(p.ucolor)"};
            const std::string source = fmt::format(
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "struct UtilParams {{ float4 color; uint4 ucolor; float4 rect; float4 uv; "
                "float depth; uint base_layer; uint2 padding; }};\n"
                "struct FragOut {{ {} c [[color({})]]; }};\n"
                "fragment FragOut {}(constant UtilParams& p [[buffer(0)]]) {{\n"
                "    FragOut o; o.c = {}; return o;\n}}\n",
                types[kind], attachment, name, values[kind]);
            library = CompileMSLLibrary(device.GetMTLDevice(), source,
                                        ShaderLanguageVersion(device), error);
        } else {
            if (!utility_library) {
                utility_library = CompileMSLLibrary(device.GetMTLDevice(), UTILITY_SOURCE,
                                                    ShaderLanguageVersion(device), error);
            }
            library = utility_library;
        }
        if (!library) {
            LOG_ERROR(Render_Metal, "Failed to compile utility shaders: {}", error);
        } else {
            function = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        }
    }
    utility_functions.emplace(key, function);
    return function;
}

id<MTLRenderPipelineState> RasterizerMetal::UtilityPipeline(const Framebuffer& framebuffer,
                                                            u32 kind, u32 attachment, u8 mask) {
    struct Key {
        std::array<u32, NUM_RT> color_formats;
        u32 depth_format;
        u32 stencil_format;
        u32 samples;
        u32 kind;
        u32 attachment;
        u32 mask;
    } key{};
    const MTLPixelFormat depth_format = framebuffer.DepthFormat();
    for (size_t i = 0; i < NUM_RT; ++i) {
        key.color_formats[i] = static_cast<u32>(framebuffer.ColorFormats()[i]);
    }
    key.depth_format = HasDepthAspect(depth_format) ? static_cast<u32>(depth_format) : 0;
    key.stencil_format = framebuffer.HasStencil() ? static_cast<u32>(depth_format) : 0;
    key.samples = (std::max)(framebuffer.Samples(), 1U);
    key.kind = kind;
    key.attachment = attachment;
    key.mask = mask;
    const u64 hash = HashBytes(key);
    const auto it = utility_pipelines.find(hash);
    if (it != utility_pipelines.end()) {
        return it->second;
    }
    id<MTLRenderPipelineState> state;
    @autoreleasepool {
        const bool is_blit = kind == UTIL_KIND_BLIT;
        id<MTLFunction> vertex =
            UtilityFunction(is_blit ? "eden_blit_vs" : "eden_clear_vs", 0, UTIL_KIND_BLIT);
        id<MTLFunction> fragment;
        if (is_blit) {
            fragment = UtilityFunction("eden_blit_fs", 0, UTIL_KIND_BLIT);
        } else if (kind <= UTIL_KIND_SINT) {
            fragment = UtilityFunction("eden_clear_fs", attachment, kind);
        }
        if (vertex && (fragment || kind == UTIL_KIND_DEPTH_STENCIL)) {
            MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
            descriptor.label = @"Eden utility pipeline";
            descriptor.vertexFunction = vertex;
            descriptor.fragmentFunction = fragment;
            for (size_t i = 0; i < NUM_RT; ++i) {
                descriptor.colorAttachments[i].pixelFormat = framebuffer.ColorFormats()[i];
                descriptor.colorAttachments[i].writeMask =
                    fragment && i == attachment ? mask : MTLColorWriteMaskNone;
            }
            descriptor.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.depth_format);
            descriptor.stencilAttachmentPixelFormat =
                static_cast<MTLPixelFormat>(key.stencil_format);
            descriptor.rasterSampleCount = key.samples;
            descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            NSError* error = nil;
            state = [device.GetMTLDevice() newRenderPipelineStateWithDescriptor:descriptor
                                                                          error:&error];
            if (!state) {
                LOG_ERROR(Render_Metal, "Failed to build utility pipeline: {}",
                          error ? error.localizedDescription.UTF8String : "unknown error");
            }
        }
    }
    utility_pipelines.emplace(hash, state);
    return state;
}

void RasterizerMetal::ClearWithDraw(const Framebuffer& framebuffer, u32 color_attachment,
                                    u8 color_mask, bool clear_depth, bool clear_stencil,
                                    MTLScissorRect rect, u32 base_layer, u32 layer_count) {
    const auto& regs = maxwell3d->regs;
    u32 kind = UTIL_KIND_DEPTH_STENCIL;
    UtilParams params{};
    if (color_mask != 0) {
        const auto format =
            VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        kind = UTIL_KIND_FLOAT;
        if (VideoCore::Surface::IsPixelFormatInteger(format)) {
            const bool is_signed = VideoCore::Surface::IsPixelFormatSignedInteger(format);
            const size_t int_size = VideoCore::Surface::PixelComponentSizeBitsInteger(format);
            kind = is_signed ? UTIL_KIND_SINT : UTIL_KIND_UINT;
            for (size_t i = 0; i < 4; ++i) {
                params.ucolor[i] =
                    is_signed ? static_cast<u32>(static_cast<s32>(
                                    f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f)))
                              : static_cast<u32>(f32(u64(int_size) << 1U) * regs.clear_color[i]);
            }
        }
        std::memcpy(params.color.data(), regs.clear_color.data(), sizeof(params.color));
    }
    params.depth = std::clamp(regs.clear_depth, 0.0f, 1.0f);
    params.base_layer = base_layer;

    id<MTLRenderPipelineState> pipeline =
        UtilityPipeline(framebuffer, kind, color_attachment, color_mask);
    if (!pipeline) {
        return;
    }
    id<MTLRenderCommandEncoder> encoder = scheduler.RenderEncoder(framebuffer.Descriptor());
    if (!encoder) {
        return;
    }
    ++stats.draw_clears;
    if (clear_depth) {
        scheduler.MarkPassDepthWrite();
    }
    if (clear_stencil) {
        scheduler.MarkPassStencilWrite();
    }
    const u64 ds_key = DEPTH_STENCIL_CUSTOM_BIT | (clear_depth ? 1 : 0) |
                       (clear_stencil ? 2 : 0) | (u64{regs.stencil_front_mask & 0xFF} << 8);
    id<MTLDepthStencilState> depth_stencil;
    if (const auto it = depth_stencil_states.find(ds_key); it != depth_stencil_states.end()) {
        depth_stencil = it->second;
    } else {
        @autoreleasepool {
            MTLDepthStencilDescriptor* descriptor = [[MTLDepthStencilDescriptor alloc] init];
            descriptor.depthCompareFunction = MTLCompareFunctionAlways;
            descriptor.depthWriteEnabled = clear_depth;
            if (clear_stencil) {
                MTLStencilDescriptor* face = [[MTLStencilDescriptor alloc] init];
                face.stencilCompareFunction = MTLCompareFunctionAlways;
                face.depthStencilPassOperation = MTLStencilOperationReplace;
                face.writeMask = regs.stencil_front_mask & 0xFF;
                descriptor.frontFaceStencil = face;
                descriptor.backFaceStencil = face;
            }
            depth_stencil = [device.GetMTLDevice() newDepthStencilStateWithDescriptor:descriptor];
        }
        depth_stencil_states.emplace(ds_key, depth_stencil);
    }
    const MTLViewport viewport{
        .originX = 0.0,
        .originY = 0.0,
        .width = static_cast<f64>(framebuffer.Width()),
        .height = static_cast<f64>(framebuffer.Height()),
        .znear = 0.0,
        .zfar = 1.0,
    };
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:viewport];
    [encoder setScissorRect:rect];
    [encoder setDepthStencilState:depth_stencil];
    [encoder setStencilReferenceValue:static_cast<u32>(regs.clear_stencil) & 0xFF];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setTriangleFillMode:MTLTriangleFillModeFill];
    [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    [encoder setDepthClipMode:MTLDepthClipModeClip];
    [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
    [encoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3
              instanceCount:(std::max)(layer_count, 1U)];
    // The next draw must reapply its own state.
    render_state = RenderEncoderState{.generation = render_state.generation};
}

void RasterizerMetal::Clear(u32 layer_count) {
    @autoreleasepool {
        FlushWork();
        std::scoped_lock lock{texture_cache.mutex};
        gpu_memory->FlushCaching();

        const auto& regs = maxwell3d->regs;
        const bool use_color = regs.clear_surface.R || regs.clear_surface.G ||
                               regs.clear_surface.B || regs.clear_surface.A;
        const bool use_depth = regs.clear_surface.Z;
        const bool use_stencil = regs.clear_surface.S;
        if (!use_color && !use_depth && !use_stencil) {
            return;
        }
        ++stats.clears;
        texture_cache.UpdateRenderTargets(true);
        const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
        if (!framebuffer || framebuffer->Width() == 0 || framebuffer->Height() == 0) {
            return;
        }
        const u32 fb_width = framebuffer->Width();
        const u32 fb_height = framebuffer->Height();
        const MTLScissorRect rect = regs.clear_control.use_scissor
                                        ? GetScissor(regs, 0, fb_width, fb_height)
                                        : ClampScissor(0, 0, fb_width, fb_height, fb_width, fb_height);
        if (rect.width == 0 || rect.height == 0) {
            return;
        }
        const u32 attachment = regs.clear_surface.RT;
        const bool clear_color = use_color && attachment < NUM_RT &&
                                 framebuffer->ColorFormats()[attachment] != MTLPixelFormatInvalid;
        const bool clear_depth = use_depth && HasDepthAspect(framebuffer->DepthFormat());
        const bool clear_stencil = use_stencil && framebuffer->HasStencil();
        if (!clear_color && !clear_depth && !clear_stencil) {
            return;
        }
        const bool full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                   regs.clear_surface.B && regs.clear_surface.A;
        const bool stencil_partial = clear_stencil && (regs.stencil_front_mask & 0xFF) != 0xFF;
        const bool full_area = rect.x == 0 && rect.y == 0 && rect.width >= fb_width &&
                               rect.height >= fb_height;
        MTLRenderPassDescriptor* const key = framebuffer->Descriptor();
        const NSUInteger fb_layers = (std::max<NSUInteger>)(key.renderTargetArrayLength, 1);
        const bool all_layers = regs.clear_surface.layer == 0 && layer_count >= fb_layers;

        if (full_area && all_layers && !stencil_partial && (!clear_color || full_channels) &&
            !scheduler.IsRenderPassActive(key)) {
            // Load action clears are free on tile based GPUs. The pass is not started here:
            // games clear several framebuffers ahead of drawing to them, and starting a pass
            // for each would end the open one every time. The clear waits for the first draw
            // on this framebuffer, or runs on its own before anything else touches it.
            MTLRenderPassDescriptor* descriptor = [key copy];
            if (clear_color) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[attachment].format);
                std::array<f64, 4> value;
                for (size_t i = 0; i < 4; ++i) {
                    value[i] = regs.clear_color[i];
                }
                if (VideoCore::Surface::IsPixelFormatInteger(format)) {
                    const bool is_signed = VideoCore::Surface::IsPixelFormatSignedInteger(format);
                    const size_t int_size = VideoCore::Surface::PixelComponentSizeBitsInteger(format);
                    for (size_t i = 0; i < 4; ++i) {
                        value[i] = is_signed ? static_cast<f64>(static_cast<s32>(
                                                   f32(s64(int_size - 1) << 1) *
                                                   (regs.clear_color[i] - 0.5f)))
                                             : static_cast<f64>(static_cast<u32>(
                                                   f32(u64(int_size) << 1U) * regs.clear_color[i]));
                    }
                }
                MTLRenderPassColorAttachmentDescriptor* color = descriptor.colorAttachments[attachment];
                color.loadAction = MTLLoadActionClear;
                color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
            }
            if (clear_depth) {
                descriptor.depthAttachment.loadAction = MTLLoadActionClear;
                descriptor.depthAttachment.clearDepth = std::clamp(regs.clear_depth, 0.0f, 1.0f);
            }
            if (clear_stencil) {
                descriptor.stencilAttachment.loadAction = MTLLoadActionClear;
                descriptor.stencilAttachment.clearStencil = static_cast<u32>(regs.clear_stencil) & 0xFF;
            }
            scheduler.DeferClear(key, descriptor, clear_depth, clear_stencil);
            return;
        }
        const u8 color_mask =
            clear_color ? static_cast<u8>((regs.clear_surface.R ? MTLColorWriteMaskRed : 0) |
                                          (regs.clear_surface.G ? MTLColorWriteMaskGreen : 0) |
                                          (regs.clear_surface.B ? MTLColorWriteMaskBlue : 0) |
                                          (regs.clear_surface.A ? MTLColorWriteMaskAlpha : 0))
                        : 0;
        ClearWithDraw(*framebuffer, clear_color ? attachment : 0, color_mask, clear_depth,
                      clear_stencil, rect, regs.clear_surface.layer, layer_count);
    }
}

void RasterizerMetal::DrawTexture() {
    @autoreleasepool {
        SCOPE_EXIT {
            gpu.TickWork();
        };
        FlushWork();

        std::scoped_lock lock{texture_cache.mutex};
        ++stats.draws;
        texture_cache.SynchronizeDescriptors(false);
        texture_cache.UpdateRenderTargets(false);

        const auto& state = maxwell3d->draw_manager.draw_texture_state;
        const auto& sampler = texture_cache.GetSampler(state.src_sampler, false);
        const auto& texture = texture_cache.GetImageView(state.src_texture);
        const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
        if (!framebuffer || framebuffer->Width() == 0 || framebuffer->Height() == 0 ||
            framebuffer->ColorFormats()[0] == MTLPixelFormatInvalid) {
            return;
        }
        id<MTLTexture> source = texture.Handle(Shader::TextureType::Color2D);
        if (!source || texture.size.width == 0 || texture.size.height == 0) {
            return;
        }
        id<MTLRenderPipelineState> pipeline =
            UtilityPipeline(*framebuffer, UTIL_KIND_BLIT, 0, MTLColorWriteMaskAll);
        if (!pipeline) {
            return;
        }
        scheduler.FlushPendingClearsFor({texture.ImageHandle()},
                                        Scheduler::ClearFlushReason::ImageOperation);
        id<MTLRenderCommandEncoder> encoder = scheduler.RenderEncoder(framebuffer->Descriptor());
        if (!encoder) {
            return;
        }
        const f32 width = static_cast<f32>(framebuffer->Width());
        const f32 height = static_cast<f32>(framebuffer->Height());
        const f32 src_width = static_cast<f32>(texture.size.width);
        const f32 src_height = static_cast<f32>(texture.size.height);
        UtilParams params{};
        params.rect = {state.dst_x0 / width * 2.0f - 1.0f, 1.0f - state.dst_y0 / height * 2.0f,
                       state.dst_x1 / width * 2.0f - 1.0f, 1.0f - state.dst_y1 / height * 2.0f};
        params.uv = {state.src_x0 / src_width, state.src_y0 / src_height, state.src_x1 / src_width,
                     state.src_y1 / src_height};

        const u64 ds_key = DEPTH_STENCIL_CUSTOM_BIT;
        id<MTLDepthStencilState> depth_stencil;
        if (const auto it = depth_stencil_states.find(ds_key); it != depth_stencil_states.end()) {
            depth_stencil = it->second;
        } else {
            @autoreleasepool {
                MTLDepthStencilDescriptor* descriptor = [[MTLDepthStencilDescriptor alloc] init];
                depth_stencil = [device.GetMTLDevice() newDepthStencilStateWithDescriptor:descriptor];
            }
            depth_stencil_states.emplace(ds_key, depth_stencil);
        }
        [encoder setRenderPipelineState:pipeline];
        [encoder setViewport:MTLViewport{0.0, 0.0, width, height, 0.0, 1.0}];
        [encoder setScissorRect:MTLScissorRect{0, 0, framebuffer->Width(), framebuffer->Height()}];
        [encoder setDepthStencilState:depth_stencil];
        [encoder setCullMode:MTLCullModeNone];
        [encoder setTriangleFillMode:MTLTriangleFillModeFill];
        [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
        [encoder setFragmentTexture:source atIndex:0];
        [encoder setFragmentSamplerState:sampler && sampler->Handle() ? sampler->Handle()
                                                                      : fallback_sampler
                                 atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        render_state = RenderEncoderState{.generation = render_state.generation};
    }
}

void RasterizerMetal::DispatchCompute() {
    @autoreleasepool {
        FlushWork(true);
        gpu_memory->FlushCaching();

        const ComputeProgram* const program = shader_frontend->CurrentComputeProgram();
        if (!program) {
            return;
        }
        ComputePipeline* const pipeline = pipeline_cache.GetComputePipeline(program->key);
        if (!pipeline || !pipeline->WaitUntilBuilt()) {
            return;
        }
        const Shader::Info& info = program->info;
        ++stats.dispatches;
        std::scoped_lock buffer_lock{buffer_cache.mutex};
        std::scoped_lock texture_lock{texture_cache.mutex};

        image_views.clear();
        samplers.clear();
        texture_cache.SynchronizeDescriptors(true);
        buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask,
                                                  &program->uniform_buffer_sizes);
        buffer_cache.UnbindComputeStorageBuffers();
        size_t ssbo_index = 0;
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                                  desc.is_written);
            ++ssbo_index;
        }
        CollectStageTextures(info, 0, true);
        texture_cache.FillImageViews(std::span(image_views.data(), image_views.size()), true, true);
        FlushPendingClearsForBoundViews();

        buffer_cache.UnbindComputeTextureBuffers();
        size_t view_index = 0;
        size_t texture_buffer_index = 0;
        const auto add_buffers = [&](const auto& descriptors, bool is_image) {
            for (const auto& desc : descriptors) {
                for (u32 i = 0; i < desc.count; ++i) {
                    ImageView& view = texture_cache.GetImageView(image_views[view_index++].id);
                    bool is_written = false;
                    if constexpr (requires { desc.is_written; }) {
                        is_written = desc.is_written;
                    }
                    buffer_cache.BindComputeTextureBuffer(texture_buffer_index++, view.GpuAddr(),
                                                          view.BufferSize(), view.format, is_written,
                                                          is_image);
                }
            }
        };
        add_buffers(info.texture_buffer_descriptors, false);
        add_buffers(info.image_buffer_descriptors, true);

        BufferCacheRuntime& runtime = buffer_cache.runtime;
        buffer_cache.UpdateComputeBuffers();
        runtime.ResetTextureBuffers();
        buffer_cache.BindHostComputeBuffers();
        const auto bound = runtime.TextureBuffers();
        compute_texture_buffers.assign(bound.begin(), bound.end());

        size_t view_cursor = 0;
        size_t sampler_cursor = 0;
        const StageBindingSource source{
            .stage = 0,
            .is_compute = true,
            .texture_buffers = compute_texture_buffers,
        };
        const ShaderModule* const module = pipeline->GetShader();
        if (!module || !EncodeStageResources(*module, info, source, view_cursor, sampler_cursor,
                                             compute_resources)) {
            return;
        }

        std::optional<std::pair<Buffer*, u32>> indirect;
        if (const auto indirect_address = kepler_compute->GetIndirectComputeAddress()) {
            static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
            indirect = buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info,
                                                 VideoCommon::ObtainBufferOperation::DiscardWrite);
        }

        id<MTLComputeCommandEncoder> encoder = scheduler.ComputeEncoder();
        if (!encoder) {
            return;
        }
        [encoder setComputePipelineState:pipeline->GetState()];
        const TranslatedShader& translated = module->GetInfo();
        if (compute_resources.argument_buffer) {
            [encoder setBuffer:compute_resources.argument_buffer
                        offset:compute_resources.argument_offset
                       atIndex:BufferSlot::DescriptorSet];
        }
        if (translated.uses_push_constants) {
            std::array<u8, 48> push_constants;
            PushConstantData(info, push_constants);
            [encoder setBytes:push_constants.data()
                       length:push_constants.size()
                      atIndex:BufferSlot::PushConstants];
        }
        if (translated.needs_buffer_size_buffer && compute_resources.sizes_buffer) {
            [encoder setBuffer:compute_resources.sizes_buffer
                        offset:compute_resources.sizes_offset
                       atIndex:BufferSlot::BufferSizes];
        }
        if (!compute_resources.reads.empty()) {
            [encoder useResources:compute_resources.reads.data()
                            count:compute_resources.reads.size()
                            usage:MTLResourceUsageRead];
        }
        if (!compute_resources.writes.empty()) {
            [encoder useResources:compute_resources.writes.data()
                            count:compute_resources.writes.size()
                            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        const MTLSize threads = pipeline->ThreadgroupSize();
        if (indirect) {
            if (indirect->first && indirect->first->Handle()) {
                [encoder dispatchThreadgroupsWithIndirectBuffer:indirect->first->Handle()
                                           indirectBufferOffset:indirect->second
                                          threadsPerThreadgroup:threads];
            }
            return;
        }
        const auto& qmd = kepler_compute->launch_description;
        const MTLSize groups = MTLSizeMake(qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z);
        if (groups.width == 0 || groups.height == 0 || groups.depth == 0) {
            return;
        }
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    }
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType) {}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32) {
    // TODO: Host queries (occlusion, streaming counters) need a query cache.
    if (!gpu_memory) {
        return;
    }
    if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
        gpu_memory->Write<u64>(gpu_addr + 8, gpu.GetTicks());
        gpu_memory->Write<u64>(gpu_addr, static_cast<u64>(payload));
    } else {
        gpu_memory->Write<u32>(gpu_addr, payload);
    }
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    // Guest threads get here for every read of cached memory, while the GPU thread holds the
    // texture cache mutex for whole draws. Only an image can be GPU modified: for a range that
    // never held one the answer of the cache, nothing to flush, is known without its mutex.
    flush_area_lookups.fetch_add(1, std::memory_order_relaxed);
    if (texture_cache_runtime.MayHoldImage(addr, size)) {
        flush_area_locks.fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lock{texture_cache.mutex};
        if (const auto area = texture_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    return VideoCore::RasterizerDownloadArea{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        shader_frontend->InvalidateRegion(addr, size);
    }
}

void RasterizerMetal::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    for (const auto& [addr, size] : sequences) {
        shader_frontend->InvalidateRegion(addr, size);
    }
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    shader_frontend->InvalidateRegion(addr, size);
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    shader_frontend->InvalidateRegion(addr, size);
    return false;
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    shader_frontend->OnCacheInvalidation(addr, size);
}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerMetal::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerMetal::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

// Metal orders every encoder after the previous ones for tracked resources, and attachments
// written by earlier draws of a pass are visible to later ones. Sampling an attachment of the
// open pass ends it through BarrierFeedbackLoop. The remaining hazard, memory written by
// shaders of the open pass, can't be ordered inside a pass on Apple GPUs: only then the pass
// is ended, every other barrier is free on tile based GPUs.
void RasterizerMetal::EndPassForShaderWrites() {
    if (scheduler.HasRenderPass() && pass_writes_generation == scheduler.EncoderGeneration()) {
        scheduler.EndEncoding(Scheduler::PassEnd::Barrier);
        ++stats.pass_barriers;
    }
}

void RasterizerMetal::WaitForIdle() {
    EndPassForShaderWrites();
    fence_manager.SignalOrdering();
}

void RasterizerMetal::FragmentBarrier() {
    EndPassForShaderWrites();
}

void RasterizerMetal::TiledCacheBarrier() {
    // Tile cache flushes have no equivalent on Metal, like on Vulkan.
}

void RasterizerMetal::FlushCommands() {
    // Like Vulkan: submissions without draws keep accumulating, fences submit on their own.
    if (draw_counter == 0) {
        return;
    }
    draw_counter = 0;
    scheduler.Flush();
}

void RasterizerMetal::FlushPendingWork() {
    draw_counter = 0;
    // Deferred clears stay pending across submissions: nothing reads a cleared texture without
    // flushing its clear first, and a clear issued at the end of a frame then folds into the
    // first pass of the next one instead of running alone here.
    if (!scheduler.IsEmpty()) {
        scheduler.Flush();
    }
}

void RasterizerMetal::FlushPendingClearsForTexture(id<MTLTexture> texture) {
    scheduler.FlushPendingClearsFor({texture}, Scheduler::ClearFlushReason::Flush);
}

void RasterizerMetal::FlushPendingClearsForBoundViews() {
    if (!scheduler.HasPendingClears()) {
        return;
    }
    for (const ImageViewInOut& in_out : image_views) {
        if (!in_out.id) {
            continue;
        }
        id<MTLTexture> handle = texture_cache.GetImageView(in_out.id).ImageHandle();
        if (scheduler.HasPendingClearFor(handle)) {
            scheduler.FlushPendingClearsFor({handle}, Scheduler::ClearFlushReason::BoundView);
            if (!scheduler.HasPendingClears()) {
                return;
            }
        }
    }
}

void RasterizerMetal::ResolveFeedbackLoop(const Framebuffer& framebuffer) {
    if (!scheduler.TakeFeedbackLoopRequest()) {
        return;
    }
    id<MTLTexture> depth_image = framebuffer.DepthImage();
    const bool wrote_depth = scheduler.PassWroteDepth();
    const bool wrote_stencil = scheduler.PassWroteStencil();
    if (!depth_image || (!wrote_depth && !wrote_stencil)) {
        return;
    }
    bool samples_depth = false;
    bool samples_stencil = false;
    for (const ImageViewInOut& in_out : image_views) {
        if (!in_out.id) {
            continue;
        }
        const ImageView& view = texture_cache.GetImageView(in_out.id);
        if (view.ImageHandle() != depth_image) {
            continue;
        }
        if (view.SamplesStencil()) {
            samples_stencil = true;
        } else {
            samples_depth = true;
        }
    }
    if ((samples_depth && wrote_depth) || (samples_stencil && wrote_stencil)) {
        scheduler.EndEncoding(Scheduler::PassEnd::FeedbackLoop);
        scheduler.CountFeedbackLoopBreak();
    }
}

void RasterizerMetal::ReportStats() {
    static constexpr auto REPORT_INTERVAL = std::chrono::seconds{30};
    const auto now = std::chrono::steady_clock::now();
    if (stats.last_report == std::chrono::steady_clock::time_point{}) {
        stats.last_report = now;
        stats.scheduler = scheduler.GetStats();
        return;
    }
    const auto elapsed = now - stats.last_report;
    if (elapsed < REPORT_INTERVAL) {
        return;
    }
    const Scheduler::Stats& current = scheduler.GetStats();
    const f64 frames = static_cast<f64>((std::max<u64>)(stats.frames, 1));
    const f64 seconds = std::chrono::duration<f64>(elapsed).count();
    const auto per_frame = [frames](u64 value) { return static_cast<f64>(value) / frames; };
    const auto delta = [&](u64 Scheduler::Stats::*member) {
        return per_frame(current.*member - stats.scheduler.*member);
    };
    const f64 timed_buffers = static_cast<f64>((std::max<u64>)(stats.gpu_timed_buffers, 1));
    const u64 flush_lookups = flush_area_lookups.exchange(0, std::memory_order_relaxed);
    const u64 flush_locks = flush_area_locks.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Render_Metal,
             "Metal stats: {} frames in {:.1f} s ({:.1f} FPS, {:.2f} dropped, {:.2f} replaced and "
             "{:.2f} presented per frame, {:.2f} drawable failures, drawable wait {:.2f} ms, "
             "throttle wait {:.2f} ms over {:.2f} waits; GPU busy {:.1f} ms per frame, "
             "start lag {:.1f} ms per command buffer); per frame: {:.0f} draws, {:.1f} "
             "dispatches, {:.1f} clears ({:.1f} deferred, {:.1f} "
             "folded into a pass, {:.1f} run alone: {:.1f} evicted, {:.1f} image ops, {:.1f} "
             "bound views, {:.1f} pass overlaps, {:.1f} flushes; {:.1f} drawn), {:.1f} render "
             "passes ({:.1f} restarts: {:.1f} after image blits, {:.1f} buffer blits, {:.1f} "
             "compute, {:.1f} clears, {:.1f} blit passes, {:.1f} barriers, {:.1f} feedback "
             "loops, {:.1f} flushes, {:.1f} other; {:.1f} ended for depth feedback, {:.1f} by "
             "barriers), {:.1f} blit encoders ({:.1f} image copies of which {:.1f} shareable, "
             "{:.1f} 2D copies, {:.1f} image uploads, {:.1f} image downloads, {:.1f} upload blits "
             "of which {:.1f} unaligned, {:.1f} capped, {:.1f} deferred uploads of which {:.1f} "
             "merged, {:.1f} extended, {:.1f} applied early, {:.1f} side copies in {:.1f} side "
             "blits with {:.1f} KiB of gap bytes, {:.1f} side patches, {:.1f} side image uploads, "
             "{:.1f} side image copies, {:.1f} side command buffers, {:.1f} buffer downloads, "
             "{:.1f} buffer fills), {:.1f} compute encoders, {:.1f} command buffers, {:.1f} "
             "pipeline lookups, {:.1f} skipped conditions, {:.2f} GPU waits ({:.2f} ms: {:.2f} "
             "buffer downloads, {:.2f} CPU reads, {:.2f} image downloads, {:.2f} unaligned "
             "copies, {:.2f} syncs), {:.1f} depth and {:.1f} stencil stores skipped, {:.1f} flush "
             "area lookups of which {:.1f} locked",
             stats.frames, seconds, frames / seconds, per_frame(stats.dropped_frames),
             per_frame(stats.replaced_frames), per_frame(stats.presented_frames),
             per_frame(stats.failed_drawables), per_frame(stats.drawable_wait_ns) / 1'000'000.0,
             per_frame(stats.throttle_wait_ns) / 1'000'000.0, per_frame(stats.throttle_waits),
             per_frame(stats.gpu_busy_ns) / 1'000'000.0,
             static_cast<f64>(stats.gpu_lag_ns) / timed_buffers / 1'000'000.0,
             per_frame(stats.draws), per_frame(stats.dispatches), per_frame(stats.clears),
             delta(&Scheduler::Stats::deferred_clears), delta(&Scheduler::Stats::applied_clears),
             delta(&Scheduler::Stats::flushed_clears),
             delta(&Scheduler::Stats::clears_run_evicted),
             delta(&Scheduler::Stats::clears_run_image_op),
             delta(&Scheduler::Stats::clears_run_bound_view),
             delta(&Scheduler::Stats::clears_run_overlap),
             delta(&Scheduler::Stats::clears_run_flush), per_frame(stats.draw_clears),
             delta(&Scheduler::Stats::render_passes), delta(&Scheduler::Stats::pass_restarts),
             delta(&Scheduler::Stats::restarts_image_blit),
             delta(&Scheduler::Stats::restarts_buffer_blit),
             delta(&Scheduler::Stats::restarts_compute), delta(&Scheduler::Stats::restarts_clear),
             delta(&Scheduler::Stats::restarts_blit_pass),
             delta(&Scheduler::Stats::restarts_barrier),
             delta(&Scheduler::Stats::restarts_feedback_loop),
             delta(&Scheduler::Stats::restarts_flush), delta(&Scheduler::Stats::restarts_other),
             delta(&Scheduler::Stats::feedback_loop_breaks), per_frame(stats.pass_barriers),
             delta(&Scheduler::Stats::blit_encoders), delta(&Scheduler::Stats::image_copies),
             delta(&Scheduler::Stats::shareable_image_copies),
             delta(&Scheduler::Stats::surface_copies), delta(&Scheduler::Stats::image_uploads),
             delta(&Scheduler::Stats::image_downloads),
             delta(&Scheduler::Stats::buffer_upload_blits),
             delta(&Scheduler::Stats::upload_blits_unaligned),
             delta(&Scheduler::Stats::upload_blits_capped),
             delta(&Scheduler::Stats::deferred_uploads), delta(&Scheduler::Stats::overlay_merges),
             delta(&Scheduler::Stats::overlay_extensions),
             delta(&Scheduler::Stats::overlay_applies), delta(&Scheduler::Stats::side_copies),
             delta(&Scheduler::Stats::side_blits),
             delta(&Scheduler::Stats::side_gap_bytes) / 1024.0,
             delta(&Scheduler::Stats::side_patches),
             delta(&Scheduler::Stats::side_image_uploads),
             delta(&Scheduler::Stats::side_image_copies),
             delta(&Scheduler::Stats::side_command_buffers),
             delta(&Scheduler::Stats::buffer_download_copies),
             delta(&Scheduler::Stats::buffer_fills), delta(&Scheduler::Stats::compute_encoders),
             delta(&Scheduler::Stats::command_buffers), per_frame(stats.pipeline_lookups),
             per_frame(stats.skipped_conditions), delta(&Scheduler::Stats::gpu_waits),
             delta(&Scheduler::Stats::gpu_wait_ns) / 1'000'000.0,
             delta(&Scheduler::Stats::waits_buffer_download),
             delta(&Scheduler::Stats::waits_cpu_read),
             delta(&Scheduler::Stats::waits_image_download),
             delta(&Scheduler::Stats::waits_unaligned_copy),
             delta(&Scheduler::Stats::waits_sync),
             delta(&Scheduler::Stats::depth_dontcare_stores),
             delta(&Scheduler::Stats::stencil_dontcare_stores), per_frame(flush_lookups),
             per_frame(flush_locks));
    buffer_cache.runtime.ReportMemoryUsage();
    stats = FrameStats{.scheduler = current, .last_report = now};
}

void RasterizerMetal::TickFrame() {
    draw_counter = 0;
    ++stats.frames;
    const Scheduler::GpuTime gpu_time = scheduler.TakeGpuTime();
    stats.gpu_busy_ns += gpu_time.busy_ns;
    stats.gpu_lag_ns += gpu_time.lag_ns;
    stats.gpu_timed_buffers += gpu_time.command_buffers;
    ReportStats();
    fence_manager.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerMetal::AccelerateConditionalRendering() {
    // Below high accuracy the report is never read and the draws run as if the condition
    // passed, the trade-off of the Vulkan query cache: reading it from guest memory downloads
    // every GPU written range of its page first, a wait for the GPU on every check. High
    // accuracy reads back reports of submitted work, but a report still in the command buffer
    // being recorded would cost a submission and a drain on every check, a frame of latency
    // each time: those run as if the condition passed as well.
    const auto& regs = maxwell3d->regs;
    if (regs.render_enable_override != Maxwell::RenderEnable::Override::UseRenderEnable) {
        return false;
    }
    switch (regs.render_enable.mode) {
    case Maxwell::RenderEnable::Mode::Conditional:
    case Maxwell::RenderEnable::Mode::IfEqual:
    case Maxwell::RenderEnable::Mode::IfNotEqual:
        break;
    default:
        return false;
    }
    if (!Settings::IsGPULevelHigh()) {
        ++stats.skipped_conditions;
        return true;
    }
    gpu_memory->FlushCaching();
    const GPUVAddr address = regs.render_enable.Address();
    constexpr u32 size = sizeof(Maxwell::ReportSemaphore::Compare);
    // Reading the report downloads every GPU written range of the pages it lies in, not just
    // the report: a page the GPU wrote anywhere costs the same wait as a GPU written report.
    const GPUVAddr page_start = Common::AlignDown(address, Core::DEVICE_PAGESIZE);
    const u64 page_size =
        Common::AlignUp(address + size, Core::DEVICE_PAGESIZE) - page_start;
    if (!gpu_memory->IsMemoryDirty(page_start, page_size, VideoCommon::CacheType::BufferCache)) {
        return false;
    }
    // Looked up with the report itself, so that no buffer gets created or grown for the
    // check; the pending writes are then tested on the part of its pages the buffer holds.
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address, size);
    if (!cpu_addr) {
        return false;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    const auto [buffer, offset] = buffer_cache.ObtainCPUBuffer(
        *cpu_addr, size, VideoCommon::ObtainBufferSynchronize::NoSynchronize,
        VideoCommon::ObtainBufferOperation::DoNothing);
    if (!buffer) {
        return false;
    }
    const DAddr buffer_begin = buffer->CpuAddr();
    const DAddr buffer_end = buffer_begin + buffer->SizeBytes();
    const DAddr pages_begin = (std::max)(buffer_begin, *cpu_addr - (address - page_start));
    const DAddr pages_end = (std::min)(buffer_end, pages_begin + page_size);
    if (pages_end <= pages_begin ||
        buffer->PendingWriteTick(pages_begin - buffer_begin, pages_end - pages_begin) <
            scheduler.CurrentTick()) {
        return false;
    }
    ++stats.skipped_conditions;
    return true;
}

bool RasterizerMetal::HasDrawTransformFeedback() {
    return false;
}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock<std::recursive_mutex> lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    shader_frontend->InvalidateRegion(*cpu_addr, copy_size);
}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock buffer_lock{buffer_cache.mutex};
        std::scoped_lock texture_lock{texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    shader_frontend->CreateChannel(channel);
    // The caches rely on the common dirty flags to notice register changes.
    VideoCommon::Dirty::SetupDirtyFlags(channel.payload->maxwell_3d.dirty.tables);
    SetupMetalDirtyFlags(channel.payload->maxwell_3d.dirty.tables);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock buffer_lock{buffer_cache.mutex};
        std::scoped_lock texture_lock{texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    shader_frontend->BindToChannel(channel_id);
    channel.payload->maxwell_3d.dirty.flags.set();
    render_state = RenderEncoderState{};
    derived = DerivedState{};
    pass_writes_generation = ~u64{0};
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock buffer_lock{buffer_cache.mutex};
        std::scoped_lock texture_lock{texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    shader_frontend->EraseChannel(channel_id);
}

void RasterizerMetal::FlushWork(bool ends_render_pass) {
    // Commits periodically so the GPU starts working before the frame ends. Committing ends
    // the open render pass, which forces tile based GPUs to store and reload every attachment,
    // so the commit waits until the guest switches render targets unless the pass gets huge.
    // Committing early lets the GPU start on a frame while the rest is recorded, and completes
    // the ticks that free buffer regions for direct CPU writes. 256 draws is a few commits per
    // frame for heavy games, each at a render target switch.
    static constexpr u32 DRAWS_TO_DISPATCH = 256;
    static constexpr u32 MAX_DEFERRED_DRAWS = DRAWS_TO_DISPATCH * 4;
    if (++draw_counter < DRAWS_TO_DISPATCH) {
        return;
    }
    const bool pass_ends = ends_render_pass || !scheduler.HasRenderPass() ||
                           (maxwell3d && maxwell3d->dirty.flags[VideoCommon::Dirty::RenderTargets]);
    if (!pass_ends && draw_counter < MAX_DEFERRED_DRAWS) {
        return;
    }
    draw_counter = 0;
    scheduler.Flush();
}

std::optional<FramebufferTextureInfo> RasterizerMetal::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return std::nullopt;
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (!image_view) {
        return std::nullopt;
    }

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.texture = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    return info;
}

} // namespace Metal
