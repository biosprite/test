// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_pipeline_cache.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "common/thread_worker.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_metal/mtl_shader.h"

namespace Metal {

class Device;

constexpr size_t NumColorAttachments = 8;
/// Matches the guest attribute count, Metal accepts indices below MTLMaxVertexAttributes (31).
constexpr size_t NumVertexAttributes = 32;

struct VertexAttributeKey {
    /// MTLVertexFormat, MTLVertexFormatInvalid for unused attributes.
    u16 format;
    u16 offset;
    /// Vertex buffer index, bound at BufferSlot::FirstVertexBuffer + buffer.
    u16 buffer;
};

struct VertexBufferKey {
    /// Zero for unused buffers.
    u32 stride;
    /// MTLVertexStepFunction.
    u32 step_function;
    u32 step_rate;
};

struct ColorAttachmentKey {
    /// MTLPixelFormat, MTLPixelFormatInvalid for unused attachments.
    u16 pixel_format;
    /// MTLColorWriteMask.
    u8 write_mask;
    u8 blend_enable;
    /// MTLBlendOperation.
    u8 rgb_blend_op;
    u8 alpha_blend_op;
    /// MTLBlendFactor.
    u8 src_rgb_factor;
    u8 dst_rgb_factor;
    u8 src_alpha_factor;
    u8 dst_alpha_factor;
};

/// State baked into a MTLRenderPipelineState. Depth/stencil, rasterizer and viewport state are
/// set on the render command encoder and are not part of the key.
struct GraphicsPipelineKey {
    /// HashSPIRV of the registered vertex shader.
    u64 vertex_shader;
    /// HashSPIRV of the registered fragment shader, zero for depth-only pipelines.
    u64 fragment_shader;
    std::array<VertexBufferKey, BufferSlot::NumVertexBuffers> vertex_buffers;
    std::array<VertexAttributeKey, NumVertexAttributes> attributes;
    std::array<ColorAttachmentKey, NumColorAttachments> color_attachments;
    /// MTLPixelFormat of the depth and stencil attachments.
    u16 depth_format;
    u16 stencil_format;
    u8 sample_count;
    /// MTLPrimitiveTopologyClass, only required for layered rendering.
    u8 topology_class;
    u8 alpha_to_coverage;
    u8 alpha_to_one;
    u8 rasterization_enabled;
    /// Keeps the key free of implicit padding so it can be hashed and compared as bytes.
    std::array<u8, 7> padding;

    [[nodiscard]] size_t Hash() const noexcept;

    [[nodiscard]] bool operator==(const GraphicsPipelineKey& rhs) const noexcept;
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineKey>);

struct ComputePipelineKey {
    /// HashSPIRV of the registered compute shader.
    u64 shader;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    [[nodiscard]] size_t Hash() const noexcept;

    [[nodiscard]] bool operator==(const ComputePipelineKey& rhs) const noexcept;
};
static_assert(std::has_unique_object_representations_v<ComputePipelineKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineKey>);

} // namespace Metal

namespace std {

template <>
struct hash<Metal::GraphicsPipelineKey> {
    size_t operator()(const Metal::GraphicsPipelineKey& key) const noexcept {
        return key.Hash();
    }
};

template <>
struct hash<Metal::ComputePipelineKey> {
    size_t operator()(const Metal::ComputePipelineKey& key) const noexcept {
        return key.Hash();
    }
};

} // namespace std

namespace Metal {

enum class PipelineStatus : u8 {
    Building,
    Ready,
    Failed,
};

/// Build state shared by graphics and compute pipelines. Pipeline objects are created by the
/// cache immediately and completed by a worker thread when shaders are compiled asynchronously.
class PipelineBase {
public:
    PipelineBase() = default;
    ~PipelineBase() = default;

    PipelineBase(const PipelineBase&) = delete;
    PipelineBase& operator=(const PipelineBase&) = delete;

    [[nodiscard]] PipelineStatus Status() const noexcept {
        return status.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsReady() const noexcept {
        return Status() == PipelineStatus::Ready;
    }

    /// Blocks until the pipeline finished building. Returns true when it can be used.
    bool WaitUntilBuilt() const;

protected:
    void Complete(PipelineStatus result);

private:
    std::atomic<PipelineStatus> status{PipelineStatus::Building};
};

class GraphicsPipeline final : public PipelineBase {
    friend class PipelineCache;

public:
    explicit GraphicsPipeline(const GraphicsPipelineKey& key);
    ~GraphicsPipeline();

    [[nodiscard]] const GraphicsPipelineKey& Key() const noexcept {
        return key;
    }

    /// Only valid once the pipeline is ready.
    [[nodiscard]] id<MTLRenderPipelineState> GetState() const noexcept {
        return state;
    }

    /// Reflection data used to encode the vertex stage argument buffer. Only valid once ready.
    [[nodiscard]] const ShaderModule* VertexShader() const noexcept {
        return vertex_shader.get();
    }

    /// Nullptr for depth-only pipelines. Only valid once ready.
    [[nodiscard]] const ShaderModule* FragmentShader() const noexcept {
        return fragment_shader.get();
    }

private:
    GraphicsPipelineKey key;
    std::shared_ptr<const ShaderModule> vertex_shader;
    std::shared_ptr<const ShaderModule> fragment_shader;
    id<MTLRenderPipelineState> state = nil;
};

class ComputePipeline final : public PipelineBase {
    friend class PipelineCache;

public:
    explicit ComputePipeline(const ComputePipelineKey& key);
    ~ComputePipeline();

    [[nodiscard]] const ComputePipelineKey& Key() const noexcept {
        return key;
    }

    /// Only valid once the pipeline is ready.
    [[nodiscard]] id<MTLComputePipelineState> GetState() const noexcept {
        return state;
    }

    /// Only valid once the pipeline is ready.
    [[nodiscard]] const ShaderModule* GetShader() const noexcept {
        return shader.get();
    }

    /// Threads per threadgroup to dispatch with, taken from the key.
    [[nodiscard]] MTLSize ThreadgroupSize() const noexcept {
        return MTLSizeMake(key.workgroup_size[0], key.workgroup_size[1], key.workgroup_size[2]);
    }

private:
    ComputePipelineKey key;
    std::shared_ptr<const ShaderModule> shader;
    id<MTLComputePipelineState> state = nil;
};

/// Owns the translated shaders and the Metal pipeline states built from them.
///
/// Shaders are registered as SPIR-V produced by Shader::Backend::SPIRV::EmitSPIRV with the
/// profile returned by GetProfile(), pipeline keys then refer to them by hash. Translation to MSL
/// and pipeline compilation happen on worker threads when asynchronous shaders are enabled.
class PipelineCache {
public:
    /// An empty binary_archive_path disables the on-disk MTLBinaryArchive.
    explicit PipelineCache(const Device& device, std::filesystem::path binary_archive_path = {});
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// Stores a SPIR-V module, translation is deferred until a pipeline uses it.
    /// Returns the hash identifying the shader in pipeline keys.
    u64 RegisterShader(ShaderStage stage, std::span<const u32> spirv);

    [[nodiscard]] bool HasShader(u64 hash) const;

    /// Returns the pipeline for the key, queuing its build when it is new. The pipeline may
    /// still be building when asynchronous shaders are enabled. Returns nullptr when the key
    /// references shaders that were not registered.
    [[nodiscard]] GraphicsPipeline* GetGraphicsPipeline(const GraphicsPipelineKey& key);

    /// Same as GetGraphicsPipeline for compute pipelines.
    [[nodiscard]] ComputePipeline* GetComputePipeline(const ComputePipelineKey& key);

    /// Blocks until every queued pipeline build finished.
    void WaitForBuilds();

    /// Writes the pipelines compiled since the last call to the binary archive on disk.
    void SerializeBinaryArchive();

    [[nodiscard]] const Shader::Profile& GetProfile() const noexcept {
        return profile;
    }

    [[nodiscard]] const Shader::HostTranslateInfo& GetHostTranslateInfo() const noexcept {
        return host_info;
    }

private:
    struct ShaderEntry {
        ShaderStage stage;
        std::vector<u32> spirv;
        std::once_flag compile_flag;
        std::shared_ptr<const ShaderModule> module;
    };

    /// Translates and compiles a registered shader the first time it is requested.
    [[nodiscard]] std::shared_ptr<const ShaderModule> GetShaderModule(u64 hash,
                                                                      ShaderStage stage);

    void BuildGraphicsPipeline(GraphicsPipeline& pipeline);
    void BuildComputePipeline(ComputePipeline& pipeline);

    void LoadBinaryArchive();

    const Device& device;
    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    bool use_asynchronous_shaders{};

    mutable std::mutex shader_mutex;
    Common::unordered_map<u64, std::unique_ptr<ShaderEntry>> shaders;

    Common::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    Common::unordered_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;

    std::mutex archive_mutex;
    std::filesystem::path archive_path;
    id<MTLBinaryArchive> binary_archive = nil;
    bool archive_dirty{};

    /// Declared last so that worker threads stop before the state they use is destroyed.
    Common::ThreadWorker workers;
};

} // namespace Metal
