// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>
#include <spirv_msl.hpp>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_shader.h"

namespace Metal {

namespace {

using spirv_cross::CompilerMSL;
using spirv_cross::SPIRType;

spv::ExecutionModel ToExecutionModel(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return spv::ExecutionModelVertex;
    case ShaderStage::Fragment:
        return spv::ExecutionModelFragment;
    case ShaderStage::Compute:
        return spv::ExecutionModelGLCompute;
    }
    throw std::invalid_argument{"Invalid shader stage"};
}

const char* EntryPointName(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return "eden_vs";
    case ShaderStage::Fragment:
        return "eden_fs";
    case ShaderStage::Compute:
        return "eden_cs";
    }
    return "eden_main";
}

struct PendingResource {
    ShaderResourceType type;
    SPIRType::BaseType basetype;
    u32 binding;
    u32 count;
};

u32 DescriptorCount(const CompilerMSL& compiler, const spirv_cross::Resource& resource) {
    const SPIRType& type = compiler.get_type(resource.type_id);
    u32 count = 1;
    for (size_t dim = 0; dim < type.array.size(); ++dim) {
        if (!type.array_size_literal[dim] || type.array[dim] == 0) {
            throw std::runtime_error{
                fmt::format("Descriptor {} has a runtime or specialized array size", resource.name)};
        }
        count *= type.array[dim];
    }
    return count;
}

void GatherResources(const CompilerMSL& compiler,
                     const spirv_cross::SmallVector<spirv_cross::Resource>& list,
                     ShaderResourceType type, SPIRType::BaseType basetype,
                     std::vector<PendingResource>& out) {
    for (const spirv_cross::Resource& resource : list) {
        const u32 set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        if (set != 0) {
            throw std::runtime_error{
                fmt::format("Descriptor {} uses unsupported set {}", resource.name, set)};
        }
        out.push_back({
            .type = type,
            .basetype = basetype,
            .binding = compiler.get_decoration(resource.id, spv::DecorationBinding),
            .count = DescriptorCount(compiler, resource),
        });
    }
}

/// Assigns argument buffer ids to every descriptor in binding order so that the layout can be
/// reproduced when encoding argument buffers. Returns the first unused id.
u32 AssignResourceIds(CompilerMSL& compiler, spv::ExecutionModel model,
                      std::vector<PendingResource>& pending,
                      std::vector<ShaderResource>& resources) {
    std::ranges::sort(pending, {}, &PendingResource::binding);
    u32 next_id = 0;
    for (const PendingResource& entry : pending) {
        ShaderResource resource{
            .type = entry.type,
            .binding = entry.binding,
            .count = entry.count,
            .id = next_id,
            .secondary_id = next_id,
        };
        next_id += entry.count;
        if (entry.type == ShaderResourceType::CombinedImageSampler ||
            entry.type == ShaderResourceType::StorageImage) {
            resource.secondary_id = next_id;
            next_id += entry.count;
        }
        spirv_cross::MSLResourceBinding binding;
        binding.stage = model;
        binding.basetype = entry.basetype;
        binding.desc_set = 0;
        binding.binding = entry.binding;
        binding.count = entry.count;
        // Emulated image atomics look up their backing buffer through msl_buffer.
        binding.msl_buffer =
            entry.type == ShaderResourceType::StorageImage ? resource.secondary_id : resource.id;
        binding.msl_texture = resource.id;
        binding.msl_sampler = resource.secondary_id;
        compiler.add_msl_resource_binding(binding);
        resources.push_back(resource);
    }
    return next_id;
}

void AddBufferBinding(CompilerMSL& compiler, spv::ExecutionModel model, u32 desc_set, u32 binding,
                      u32 msl_buffer) {
    spirv_cross::MSLResourceBinding resource_binding;
    resource_binding.stage = model;
    resource_binding.basetype = SPIRType::Struct;
    resource_binding.desc_set = desc_set;
    resource_binding.binding = binding;
    resource_binding.count = 1;
    resource_binding.msl_buffer = msl_buffer;
    compiler.add_msl_resource_binding(resource_binding);
}

void ConfigureCompiler(CompilerMSL& compiler, const ShaderTranslateOptions& options) {
    CompilerMSL::Options msl = compiler.get_msl_options();
    msl.platform = CompilerMSL::Options::macOS;
    msl.set_msl_version(options.msl_major, options.msl_minor);
    msl.argument_buffers = true;
    msl.argument_buffers_tier = options.argument_buffers_tier2
                                    ? CompilerMSL::Options::ArgumentBuffersTier::Tier2
                                    : CompilerMSL::Options::ArgumentBuffersTier::Tier1;
    // Keep unused descriptors in the argument buffer so its layout only depends on reflection.
    msl.force_active_argument_buffer_resources = true;
    msl.buffer_size_buffer_index = BufferSlot::BufferSizes;
    // The texture cache stores 1D images as 2D textures and texel buffers as texture buffers.
    msl.texture_1D_as_2D = true;
    msl.texture_buffer_native = true;
    // Guest shaders may write fewer components than the render target format has.
    msl.pad_fragment_output_components = true;
    msl.enable_point_size_default = false;
    msl.default_point_size = 1.0f;
    msl.agx_manual_cube_grad_fixup = options.apple_gpu;
    msl.sample_dref_lod_array_as_grad = options.apple_gpu;
    compiler.set_msl_options(msl);

    spirv_cross::CompilerGLSL::Options common = compiler.get_common_options();
    // Shaders are generated for Vulkan clip space, flip Y so Metal rasterizes them the same way.
    common.vertex.flip_vert_y = options.stage == ShaderStage::Vertex;
    compiler.set_common_options(common);
}

TranslatedShader Translate(std::span<const u32> spirv, const ShaderTranslateOptions& options) {
    CompilerMSL compiler(spirv.data(), spirv.size());
    const spv::ExecutionModel model = ToExecutionModel(options.stage);

    const auto entry_points = compiler.get_entry_points_and_stages();
    const auto entry_it = std::ranges::find(entry_points, model, &spirv_cross::EntryPoint::execution_model);
    if (entry_it == entry_points.end()) {
        throw std::runtime_error{"Module has no entry point for the requested stage"};
    }
    const std::string entry_name{EntryPointName(options.stage)};
    compiler.set_entry_point(entry_it->name, model);
    compiler.rename_entry_point(entry_it->name, entry_name, model);

    ConfigureCompiler(compiler, options);

    TranslatedShader result{};
    result.stage = options.stage;

    const spirv_cross::ShaderResources shader_resources = compiler.get_shader_resources();
    if (!shader_resources.subpass_inputs.empty() || !shader_resources.atomic_counters.empty() ||
        !shader_resources.acceleration_structures.empty()) {
        throw std::runtime_error{"Module uses resource types unsupported by the Metal backend"};
    }
    std::vector<PendingResource> pending;
    GatherResources(compiler, shader_resources.uniform_buffers, ShaderResourceType::UniformBuffer,
                    SPIRType::Struct, pending);
    GatherResources(compiler, shader_resources.storage_buffers, ShaderResourceType::StorageBuffer,
                    SPIRType::Struct, pending);
    GatherResources(compiler, shader_resources.sampled_images,
                    ShaderResourceType::CombinedImageSampler, SPIRType::SampledImage, pending);
    GatherResources(compiler, shader_resources.separate_images, ShaderResourceType::SampledImage,
                    SPIRType::Image, pending);
    GatherResources(compiler, shader_resources.separate_samplers, ShaderResourceType::Sampler,
                    SPIRType::Sampler, pending);
    GatherResources(compiler, shader_resources.storage_images, ShaderResourceType::StorageImage,
                    SPIRType::Image, pending);

    const u32 buffer_sizes_id = AssignResourceIds(compiler, model, pending, result.resources);
    AddBufferBinding(compiler, model, 0, spirv_cross::kArgumentBufferBinding,
                     BufferSlot::DescriptorSet);
    AddBufferBinding(compiler, model, 0, spirv_cross::kBufferSizeBufferBinding, buffer_sizes_id);
    AddBufferBinding(compiler, model, spirv_cross::kPushConstDescSet,
                     spirv_cross::kPushConstBinding, BufferSlot::PushConstants);

    result.source = compiler.compile();
    result.entry_point = compiler.get_cleansed_entry_point_name(entry_name, model);
    result.uses_push_constants = !shader_resources.push_constant_buffers.empty();
    result.needs_buffer_size_buffer = compiler.needs_buffer_size_buffer();
    result.argument_buffer_length = buffer_sizes_id;
    if (result.needs_buffer_size_buffer) {
        result.buffer_sizes_id = buffer_sizes_id;
        result.argument_buffer_length = buffer_sizes_id + 1;
    }
    if (options.stage == ShaderStage::Compute) {
        for (u32 dim = 0; dim < 3; ++dim) {
            result.workgroup_size[dim] =
                compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, dim);
        }
    }
    return result;
}

} // Anonymous namespace

ShaderTranslateOptions MakeShaderTranslateOptions(const Device& device, ShaderStage stage) {
    const DeviceFeatures& features = device.GetFeatures();
    // MTLLanguageVersion encodes the major version in the upper 16 bits.
    const auto version = static_cast<u32>(ShaderLanguageVersion(device));
    return ShaderTranslateOptions{
        .stage = stage,
        .msl_major = version >> 16,
        .msl_minor = version & 0xffff,
        .argument_buffers_tier2 = features.argument_buffers_tier2,
        .apple_gpu = features.apple_gpu_family != 0,
    };
}

std::optional<TranslatedShader> TranslateSPIRVToMSL(std::span<const u32> spirv,
                                                    const ShaderTranslateOptions& options) {
    try {
        return Translate(spirv, options);
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Metal, "SPIR-V to MSL translation failed: {}", e.what());
        return std::nullopt;
    }
}

MTLLanguageVersion ShaderLanguageVersion(const Device& device) {
    if (!device.GetFeatures().metal3) {
        return MTLLanguageVersion2_4;
    }
#if defined(__MAC_14_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_14_0
    // MSL 3.1 adds native texture atomics, older versions emulate them through buffers.
    if (@available(macOS 14.0, *)) {
        return MTLLanguageVersion3_1;
    }
#endif
    return MTLLanguageVersion3_0;
}

id<MTLLibrary> CompileMSLLibrary(id<MTLDevice> device, std::string_view source,
                                 MTLLanguageVersion language_version, std::string& error) {
    @autoreleasepool {
        NSString* const ns_source = [[NSString alloc] initWithBytes:source.data()
                                                             length:source.size()
                                                           encoding:NSUTF8StringEncoding];
        if (!ns_source) {
            error = "MSL source is not valid UTF-8";
            return nil;
        }
        MTLCompileOptions* const options = [[MTLCompileOptions alloc] init];
        options.languageVersion = language_version;
        // Guest shaders rely on invariant positions across pipelines (depth prepasses).
        options.preserveInvariance = YES;
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else
#endif
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }

        NSError* ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:ns_source
                                                      options:options
                                                        error:&ns_error];
        if (!library) {
            error = ns_error ? std::string{ns_error.localizedDescription.UTF8String}
                             : std::string{"Unknown error"};
        }
        return library;
    }
}

u64 HashSPIRV(std::span<const u32> spirv) {
    return Common::CityHash64(reinterpret_cast<const char*>(spirv.data()), spirv.size_bytes());
}

std::shared_ptr<const ShaderModule> ShaderModule::Create(const Device& device,
                                                         std::span<const u32> spirv,
                                                         ShaderStage stage,
                                                         std::string_view label) {
    std::optional<TranslatedShader> translated =
        TranslateSPIRVToMSL(spirv, MakeShaderTranslateOptions(device, stage));
    if (!translated) {
        return nullptr;
    }
    @autoreleasepool {
        std::string error;
        id<MTLLibrary> library = CompileMSLLibrary(device.GetMTLDevice(), translated->source,
                                                   ShaderLanguageVersion(device), error);
        if (!library) {
            LOG_ERROR(Render_Metal, "Failed to compile MSL for {}: {}", label, error);
            LOG_DEBUG(Render_Metal, "MSL source:\n{}", translated->source);
            return nullptr;
        }
        NSString* const entry_point =
            [NSString stringWithUTF8String:translated->entry_point.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:entry_point];
        if (!function) {
            LOG_ERROR(Render_Metal, "MSL library for {} has no function {}", label,
                      translated->entry_point);
            return nullptr;
        }
        if (!label.empty()) {
            NSString* const ns_label = [[NSString alloc] initWithBytes:label.data()
                                                                length:label.size()
                                                              encoding:NSUTF8StringEncoding];
            library.label = ns_label;
            function.label = ns_label;
        }
        return std::make_shared<const ShaderModule>(std::move(*translated), library, function,
                                                    HashSPIRV(spirv));
    }
}

ShaderModule::ShaderModule(TranslatedShader translated, id<MTLLibrary> library_,
                           id<MTLFunction> function_, u64 hash_)
    : info{std::move(translated)}, library{library_}, function{function_}, hash{hash_} {}

ShaderModule::~ShaderModule() = default;

} // namespace Metal
