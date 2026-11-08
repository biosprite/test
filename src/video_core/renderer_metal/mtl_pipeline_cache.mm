// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <system_error>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/stage.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"

namespace Metal {

namespace {

constexpr u32 MaxMetalVertexAttributes = 31;

constexpr u32 StageBit(Shader::Stage stage) {
    return 1U << static_cast<u32>(stage);
}

Shader::Profile MakeProfile(const Device& device) {
    const DeviceFeatures& features = device.GetFeatures();
    const bool apple_gpu = features.apple_gpu_family != 0;
    const bool tier2 = features.argument_buffers_tier2;
    return Shader::Profile{
        .supported_spirv = 0x00010300,
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = false,
        .support_int8 = true,
        .support_int16 = true,
        .support_int64 = features.metal3,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_quad_shuffles = true,
        .support_vote = true,
        // SIMD-group functions are only guaranteed in fragment and kernel functions.
        .supported_subgroup_stages = StageBit(Shader::Stage::Fragment) |
                                     StageBit(Shader::Stage::Compute),
        .support_viewport_index_layer_non_geometry = true,
        .support_viewport_mask = false,
        .support_typeless_image_loads = true,
        .support_demote_to_helper_invocation = true,
        .support_int64_atomics = false,
        .support_shared_int64_atomics = false,
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = false,
        .support_native_ndc = false,
        .support_scaled_attributes = false,
        .support_multi_viewport = true,
        .support_geometry_streams = false,
        .support_sampled_image_array_nonuniform_indexing = tier2,
        .support_storage_image_array_nonuniform_indexing = tier2,
        .support_uniform_texel_buffer_array_nonuniform_indexing = tier2,
        .support_storage_texel_buffer_array_nonuniform_indexing = tier2,
        // AMD GPUs in Intel Macs execute 64 wide SIMD-groups.
        .warp_size_potentially_larger_than_guest = !apple_gpu,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_fastmath_off = true,
        .max_subgroup_size = apple_gpu ? 32U : 64U,
        .min_ssbo_alignment = 16,
        .max_user_clip_distances = 8,
    };
}

Shader::HostTranslateInfo MakeHostTranslateInfo(const Device& device) {
    const DeviceFeatures& features = device.GetFeatures();
    Shader::HostTranslateInfo info{
        .min_ssbo_alignment = 16,
        .support_float64 = false,
        .support_float16 = true,
        .support_int64 = features.metal3,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = true,
        .support_viewport_mask = false,
        // The shader frontend enables it for geometry programs it folds into the vertex stage.
        .support_geometry_shader_passthrough = false,
        .support_conditional_barrier = false,
        .single_lane_geometry_subgroups = true,
        .support_attribute_texture_handle = true,
    };
    if (!features.argument_buffers_tier2) {
        // Tier 1 argument buffer limits on macOS.
        info.max_per_stage_descriptor_sampled_images = 128;
        info.max_descriptor_set_sampled_images = 128;
        info.max_descriptor_set_storage_images = 8;
        info.max_descriptor_set_samplers = 16;
        info.max_descriptor_set_uniform_buffers = 31;
        info.max_descriptor_set_storage_buffers = 31;
    }
    // Remaining limits fall back to the default descriptor limit.
    info.ApplyDescriptorLimitPolicy();
    return info;
}

NSString* MakeLabel(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

std::string ErrorString(NSError* error) {
    return error ? std::string{error.localizedDescription.UTF8String}
                 : std::string{"Unknown error"};
}

MTLVertexDescriptor* MakeVertexDescriptor(const GraphicsPipelineKey& key) {
    MTLVertexDescriptor* const descriptor = [MTLVertexDescriptor vertexDescriptor];
    std::array<bool, BufferSlot::NumVertexBuffers> used_buffers{};
    for (u32 index = 0; index < NumVertexAttributes; ++index) {
        const VertexAttributeKey& attribute = key.attributes[index];
        if (attribute.format == MTLVertexFormatInvalid) {
            continue;
        }
        if (index >= MaxMetalVertexAttributes || attribute.buffer >= BufferSlot::NumVertexBuffers) {
            LOG_WARNING(Render_Metal, "Vertex attribute {} on buffer {} can't be represented",
                        index, attribute.buffer);
            continue;
        }
        MTLVertexAttributeDescriptor* const desc = descriptor.attributes[index];
        desc.format = static_cast<MTLVertexFormat>(attribute.format);
        desc.offset = attribute.offset;
        desc.bufferIndex = BufferSlot::FirstVertexBuffer + attribute.buffer;
        used_buffers[attribute.buffer] = true;
    }
    for (u32 index = 0; index < BufferSlot::NumVertexBuffers; ++index) {
        if (!used_buffers[index]) {
            continue;
        }
        const VertexBufferKey& buffer = key.vertex_buffers[index];
        MTLVertexBufferLayoutDescriptor* const layout =
            descriptor.layouts[BufferSlot::FirstVertexBuffer + index];
        layout.stride = buffer.stride;
        layout.stepFunction = static_cast<MTLVertexStepFunction>(buffer.step_function);
        layout.stepRate = buffer.step_rate;
    }
    return descriptor;
}

void ConfigureAttachments(MTLRenderPipelineDescriptor* descriptor,
                          const GraphicsPipelineKey& key) {
    for (u32 index = 0; index < NumColorAttachments; ++index) {
        const ColorAttachmentKey& color = key.color_attachments[index];
        MTLRenderPipelineColorAttachmentDescriptor* const desc =
            descriptor.colorAttachments[index];
        desc.pixelFormat = static_cast<MTLPixelFormat>(color.pixel_format);
        if (color.pixel_format == MTLPixelFormatInvalid) {
            continue;
        }
        desc.writeMask = static_cast<MTLColorWriteMask>(color.write_mask);
        desc.blendingEnabled = color.blend_enable != 0;
        desc.rgbBlendOperation = static_cast<MTLBlendOperation>(color.rgb_blend_op);
        desc.alphaBlendOperation = static_cast<MTLBlendOperation>(color.alpha_blend_op);
        desc.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(color.src_rgb_factor);
        desc.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(color.dst_rgb_factor);
        desc.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(color.src_alpha_factor);
        desc.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(color.dst_alpha_factor);
    }
    descriptor.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.depth_format);
    descriptor.stencilAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.stencil_format);
    descriptor.rasterSampleCount = std::max<NSUInteger>(key.sample_count, 1);
    descriptor.inputPrimitiveTopology = static_cast<MTLPrimitiveTopologyClass>(key.topology_class);
    descriptor.alphaToCoverageEnabled = key.alpha_to_coverage != 0;
    descriptor.alphaToOneEnabled = key.alpha_to_one != 0;
    descriptor.rasterizationEnabled = key.rasterization_enabled != 0;
}

} // Anonymous namespace

size_t GraphicsPipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& rhs) const noexcept {
    return std::memcmp(this, &rhs, sizeof *this) == 0;
}

size_t ComputePipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

bool ComputePipelineKey::operator==(const ComputePipelineKey& rhs) const noexcept {
    return std::memcmp(this, &rhs, sizeof *this) == 0;
}

bool PipelineBase::WaitUntilBuilt() const {
    status.wait(PipelineStatus::Building, std::memory_order_acquire);
    return IsReady();
}

void PipelineBase::Complete(PipelineStatus result) {
    status.store(result, std::memory_order_release);
    status.notify_all();
}

GraphicsPipeline::GraphicsPipeline(const GraphicsPipelineKey& key_) : key{key_} {}

GraphicsPipeline::~GraphicsPipeline() = default;

ComputePipeline::ComputePipeline(const ComputePipelineKey& key_) : key{key_} {}

ComputePipeline::~ComputePipeline() = default;

PipelineCache::PipelineCache(const Device& device_, std::filesystem::path binary_archive_path)
    : device{device_}, profile{MakeProfile(device_)}, host_info{MakeHostTranslateInfo(device_)},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      archive_path{std::move(binary_archive_path)},
      workers{std::max(std::thread::hardware_concurrency(), 2U) - 1, "MetalPipelineBuilder"} {
    if (!archive_path.empty()) {
        LoadBinaryArchive();
    }
}

PipelineCache::~PipelineCache() {
    workers.WaitForRequests();
    SerializeBinaryArchive();
}

u64 PipelineCache::RegisterShader(ShaderStage stage, std::span<const u32> spirv) {
    const u64 hash = HashSPIRV(spirv);
    std::scoped_lock lock{shader_mutex};
    auto [it, inserted] = shaders.try_emplace(hash);
    if (inserted) {
        it->second = std::make_unique<ShaderEntry>();
        it->second->stage = stage;
        it->second->spirv.assign(spirv.begin(), spirv.end());
    }
    return hash;
}

bool PipelineCache::HasShader(u64 hash) const {
    std::scoped_lock lock{shader_mutex};
    return shaders.contains(hash);
}

std::shared_ptr<const ShaderModule> PipelineCache::GetShaderModule(u64 hash, ShaderStage stage) {
    ShaderEntry* entry;
    {
        std::scoped_lock lock{shader_mutex};
        const auto it = shaders.find(hash);
        if (it == shaders.end()) {
            return nullptr;
        }
        entry = it->second.get();
    }
    if (entry->stage != stage) {
        LOG_ERROR(Render_Metal, "Shader {:016x} used with the wrong stage", hash);
        return nullptr;
    }
    // Entries are never removed, so the pointer stays valid outside of the lock.
    std::call_once(entry->compile_flag, [&] {
        entry->module =
            ShaderModule::Create(device, entry->spirv, stage, fmt::format("Shader {:016x}", hash));
        // The SPIR-V is only kept to compile the module once.
        entry->spirv = {};
    });
    return entry->module;
}

GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const GraphicsPipelineKey& key) {
    const auto it = graphics_cache.find(key);
    if (it != graphics_cache.end()) {
        return it->second.get();
    }
    if (!HasShader(key.vertex_shader) ||
        (key.fragment_shader != 0 && !HasShader(key.fragment_shader))) {
        return nullptr;
    }
    auto pipeline = std::make_unique<GraphicsPipeline>(key);
    GraphicsPipeline* const raw = pipeline.get();
    graphics_cache.emplace(key, std::move(pipeline));
    if (use_asynchronous_shaders) {
        workers.QueueWork([this, raw] { BuildGraphicsPipeline(*raw); });
    } else {
        BuildGraphicsPipeline(*raw);
    }
    return raw;
}

ComputePipeline* PipelineCache::GetComputePipeline(const ComputePipelineKey& key) {
    const auto it = compute_cache.find(key);
    if (it != compute_cache.end()) {
        return it->second.get();
    }
    if (!HasShader(key.shader)) {
        return nullptr;
    }
    auto pipeline = std::make_unique<ComputePipeline>(key);
    ComputePipeline* const raw = pipeline.get();
    compute_cache.emplace(key, std::move(pipeline));
    if (use_asynchronous_shaders) {
        workers.QueueWork([this, raw] { BuildComputePipeline(*raw); });
    } else {
        BuildComputePipeline(*raw);
    }
    return raw;
}

void PipelineCache::WaitForBuilds() {
    workers.WaitForRequests();
}

void PipelineCache::BuildGraphicsPipeline(GraphicsPipeline& pipeline) {
    @autoreleasepool {
        const GraphicsPipelineKey& key = pipeline.key;
        pipeline.vertex_shader = GetShaderModule(key.vertex_shader, ShaderStage::Vertex);
        if (key.fragment_shader != 0) {
            pipeline.fragment_shader = GetShaderModule(key.fragment_shader, ShaderStage::Fragment);
        }
        if (!pipeline.vertex_shader || (key.fragment_shader != 0 && !pipeline.fragment_shader)) {
            pipeline.Complete(PipelineStatus::Failed);
            return;
        }

        MTLRenderPipelineDescriptor* const descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.label = MakeLabel(fmt::format("Graphics {:016x} {:016x}", key.vertex_shader,
                                                 key.fragment_shader));
        descriptor.vertexFunction = pipeline.vertex_shader->GetFunction();
        descriptor.fragmentFunction =
            pipeline.fragment_shader ? pipeline.fragment_shader->GetFunction() : nil;
        descriptor.vertexDescriptor = MakeVertexDescriptor(key);
        ConfigureAttachments(descriptor, key);

        id<MTLBinaryArchive> archive;
        {
            std::scoped_lock lock{archive_mutex};
            archive = binary_archive;
        }
        if (archive) {
            descriptor.binaryArchives = @[ archive ];
        }

        NSError* error = nil;
        id<MTLRenderPipelineState> state =
            [device.GetMTLDevice() newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!state) {
            LOG_ERROR(Render_Metal, "Failed to build graphics pipeline {:016x}/{:016x}: {}",
                      key.vertex_shader, key.fragment_shader, ErrorString(error));
            pipeline.Complete(PipelineStatus::Failed);
            return;
        }
        if (archive) {
            std::scoped_lock lock{archive_mutex};
            NSError* archive_error = nil;
            if ([archive addRenderPipelineFunctionsWithDescriptor:descriptor
                                                            error:&archive_error]) {
                archive_dirty = true;
            } else {
                LOG_DEBUG(Render_Metal, "Binary archive rejected graphics pipeline: {}",
                          ErrorString(archive_error));
            }
        }
        pipeline.state = state;
        pipeline.Complete(PipelineStatus::Ready);
    }
}

void PipelineCache::BuildComputePipeline(ComputePipeline& pipeline) {
    @autoreleasepool {
        const ComputePipelineKey& key = pipeline.key;
        pipeline.shader = GetShaderModule(key.shader, ShaderStage::Compute);
        if (!pipeline.shader) {
            pipeline.Complete(PipelineStatus::Failed);
            return;
        }

        MTLComputePipelineDescriptor* const descriptor =
            [[MTLComputePipelineDescriptor alloc] init];
        descriptor.label = MakeLabel(fmt::format("Compute {:016x}", key.shader));
        descriptor.computeFunction = pipeline.shader->GetFunction();

        id<MTLBinaryArchive> archive;
        {
            std::scoped_lock lock{archive_mutex};
            archive = binary_archive;
        }
        if (archive) {
            descriptor.binaryArchives = @[ archive ];
        }

        NSError* error = nil;
        id<MTLComputePipelineState> state =
            [device.GetMTLDevice() newComputePipelineStateWithDescriptor:descriptor
                                                                 options:MTLPipelineOptionNone
                                                              reflection:nil
                                                                   error:&error];
        if (!state) {
            LOG_ERROR(Render_Metal, "Failed to build compute pipeline {:016x}: {}", key.shader,
                      ErrorString(error));
            pipeline.Complete(PipelineStatus::Failed);
            return;
        }
        const u64 threads = u64{key.workgroup_size[0]} * key.workgroup_size[1] *
                            key.workgroup_size[2];
        if (threads > state.maxTotalThreadsPerThreadgroup) {
            LOG_ERROR(Render_Metal, "Compute pipeline {:016x} needs {} threads, device allows {}",
                      key.shader, threads, state.maxTotalThreadsPerThreadgroup);
            pipeline.Complete(PipelineStatus::Failed);
            return;
        }
        if (archive) {
            std::scoped_lock lock{archive_mutex};
            NSError* archive_error = nil;
            if ([archive addComputePipelineFunctionsWithDescriptor:descriptor
                                                             error:&archive_error]) {
                archive_dirty = true;
            } else {
                LOG_DEBUG(Render_Metal, "Binary archive rejected compute pipeline: {}",
                          ErrorString(archive_error));
            }
        }
        pipeline.state = state;
        pipeline.Complete(PipelineStatus::Ready);
    }
}

void PipelineCache::LoadBinaryArchive() {
    @autoreleasepool {
        id<MTLDevice> mtl_device = device.GetMTLDevice();
        NSString* const path = MakeLabel(archive_path.string());
        NSURL* const url = [NSURL fileURLWithPath:path];

        std::error_code ec;
        MTLBinaryArchiveDescriptor* const descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
        if (std::filesystem::exists(archive_path, ec)) {
            descriptor.url = url;
        }
        NSError* error = nil;
        id<MTLBinaryArchive> archive = [mtl_device newBinaryArchiveWithDescriptor:descriptor
                                                                            error:&error];
        if (!archive && descriptor.url) {
            // Archives become unusable after OS or driver updates, start a new one.
            LOG_WARNING(Render_Metal, "Discarding binary archive {}: {}", archive_path.string(),
                        ErrorString(error));
            descriptor.url = nil;
            error = nil;
            archive = [mtl_device newBinaryArchiveWithDescriptor:descriptor error:&error];
        }
        if (!archive) {
            LOG_ERROR(Render_Metal, "Failed to create binary archive: {}", ErrorString(error));
            return;
        }
        archive.label = @"Eden pipeline archive";
        std::scoped_lock lock{archive_mutex};
        binary_archive = archive;
    }
}

void PipelineCache::SerializeBinaryArchive() {
    std::scoped_lock lock{archive_mutex};
    if (!binary_archive || !archive_dirty) {
        return;
    }
    @autoreleasepool {
        std::error_code ec;
        std::filesystem::create_directories(archive_path.parent_path(), ec);
        NSURL* const url = [NSURL fileURLWithPath:MakeLabel(archive_path.string())];
        NSError* error = nil;
        if ([binary_archive serializeToURL:url error:&error]) {
            archive_dirty = false;
        } else {
            LOG_ERROR(Render_Metal, "Failed to write binary archive {}: {}",
                      archive_path.string(), ErrorString(error));
        }
    }
}

} // namespace Metal
