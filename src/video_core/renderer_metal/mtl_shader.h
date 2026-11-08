// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_shader.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"

namespace Metal {

class Device;

enum class ShaderStage : u8 {
    Vertex,
    Fragment,
    Compute,
};

/// Buffer argument table slots shared by every translated shader. Slots 20 and above are left to
/// the SPIRV-Cross auxiliary buffers of features the backend does not enable.
namespace BufferSlot {
/// Argument buffer holding the resources of descriptor set 0.
constexpr u32 DescriptorSet = 0;
/// Push constant block (rescaling and render area data).
constexpr u32 PushConstants = 1;
/// Storage buffer sizes used by OpArrayLength outside of argument buffers.
constexpr u32 BufferSizes = 2;
/// First vertex buffer slot, vertex buffer N is bound at FirstVertexBuffer + N.
constexpr u32 FirstVertexBuffer = 4;
constexpr u32 NumVertexBuffers = 16;
} // namespace BufferSlot

enum class ShaderResourceType : u8 {
    UniformBuffer,
    StorageBuffer,
    /// Texture and sampler pair, the sampler ids follow the texture ids.
    CombinedImageSampler,
    /// Sampled texture or texel buffer without a sampler.
    SampledImage,
    Sampler,
    /// Storage image or storage texel buffer.
    StorageImage,
};

/// Location of a SPIR-V descriptor inside the argument buffer of a translated shader.
struct ShaderResource {
    ShaderResourceType type;
    u32 binding;
    /// Number of array elements, every element takes its own argument buffer id.
    u32 count;
    /// First [[id(n)]] of the buffer or texture.
    u32 id;
    /// First [[id(n)]] of the sampler of a combined image sampler, or of the buffer backing
    /// emulated atomics of a storage image (MSL older than 3.1). Equal to id otherwise.
    u32 secondary_id;
};

/// Parameters controlling the SPIR-V to MSL translation.
struct ShaderTranslateOptions {
    ShaderStage stage{};
    u32 msl_major{3};
    u32 msl_minor{0};
    bool argument_buffers_tier2{};
    /// Enables workarounds for Apple GPU (AGX) specific behaviour.
    bool apple_gpu{};
};

/// MSL source and reflection data produced from a SPIR-V module.
struct TranslatedShader {
    std::string source;
    std::string entry_point;
    ShaderStage stage{};
    std::vector<ShaderResource> resources;
    /// Argument buffer id of a `constant uint*` pointing at the storage buffer sizes, indexed by
    /// the storage buffer ids. Only present when the shader uses OpArrayLength.
    std::optional<u32> buffer_sizes_id;
    /// Total number of argument buffer ids used by the shader.
    u32 argument_buffer_length{};
    bool uses_push_constants{};
    /// The entry point also declares the buffer sizes at BufferSlot::BufferSizes, so a buffer
    /// must be bound there even though the argument buffer copy is the one being read.
    bool needs_buffer_size_buffer{};
    /// Local workgroup size of compute shaders.
    std::array<u32, 3> workgroup_size{};
};

[[nodiscard]] ShaderTranslateOptions MakeShaderTranslateOptions(const Device& device,
                                                                ShaderStage stage);

/// Translates a SPIR-V module to MSL. Returns std::nullopt and logs the reason on failure.
[[nodiscard]] std::optional<TranslatedShader> TranslateSPIRVToMSL(
    std::span<const u32> spirv, const ShaderTranslateOptions& options);

/// Language version used to compile translated shaders on this device.
[[nodiscard]] MTLLanguageVersion ShaderLanguageVersion(const Device& device);

/// Compiles MSL source code to a Metal library. Returns nil and fills error on failure.
[[nodiscard]] id<MTLLibrary> CompileMSLLibrary(id<MTLDevice> device, std::string_view source,
                                               MTLLanguageVersion language_version,
                                               std::string& error);

/// Hashes a SPIR-V module, the value identifies shaders in pipeline keys.
[[nodiscard]] u64 HashSPIRV(std::span<const u32> spirv);

/// Translated and compiled shader function together with the reflection data needed to bind its
/// resources. Immutable once created, so it can be shared between threads and pipelines.
class ShaderModule {
public:
    /// Translates and compiles the given SPIR-V. Returns nullptr and logs the reason on failure.
    [[nodiscard]] static std::shared_ptr<const ShaderModule> Create(const Device& device,
                                                                    std::span<const u32> spirv,
                                                                    ShaderStage stage,
                                                                    std::string_view label);

    explicit ShaderModule(TranslatedShader translated, id<MTLLibrary> library,
                          id<MTLFunction> function, u64 hash);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    [[nodiscard]] id<MTLFunction> GetFunction() const noexcept {
        return function;
    }

    [[nodiscard]] id<MTLLibrary> GetLibrary() const noexcept {
        return library;
    }

    [[nodiscard]] const TranslatedShader& GetInfo() const noexcept {
        return info;
    }

    [[nodiscard]] ShaderStage GetStage() const noexcept {
        return info.stage;
    }

    [[nodiscard]] u64 Hash() const noexcept {
        return hash;
    }

private:
    TranslatedShader info;
    id<MTLLibrary> library;
    id<MTLFunction> function;
    u64 hash;
};

} // namespace Metal
