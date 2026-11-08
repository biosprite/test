// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>

#import <Foundation/Foundation.h>

#include "common/literals.h"
#include "common/logging.h"
#include "video_core/renderer_metal/mtl_device.h"

namespace Metal {

namespace {

u32 HighestAppleFamily(id<MTLDevice> device) {
    // MTLGPUFamilyApple1 == 1001. Probing raw values keeps this working with SDKs that predate
    // newer families, unknown families are simply reported as unsupported.
    constexpr u32 apple_family_base = 1000;
    constexpr u32 max_probed_family = 12;
    u32 highest = 0;
    for (u32 family = 1; family <= max_probed_family; ++family) {
        if ([device supportsFamily:static_cast<MTLGPUFamily>(apple_family_base + family)]) {
            highest = family;
        }
    }
    return highest;
}

} // Anonymous namespace

Device::Device() {
    // Both calls return retained objects.
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error{"No Metal device available"};
    }
    queue = [device newCommandQueue];
    if (!queue) {
        throw std::runtime_error{"Failed to create Metal command queue"};
    }

    @autoreleasepool {
        queue.label = @"Eden cache queue";

        name = device.name ? std::string{device.name.UTF8String} : std::string{"Unknown"};
        recommended_working_set_size = device.recommendedMaxWorkingSetSize;

        features.apple_gpu_family = HighestAppleFamily(device);
        features.metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
        features.argument_buffers_tier2 = device.argumentBuffersSupport == MTLArgumentBuffersTier2;
        features.unified_memory = device.hasUnifiedMemory;
        features.raytracing = device.supportsRaytracing;
        features.bc_texture_compression = device.supportsBCTextureCompression;
        // Packed 16-bit formats are only available on Apple GPUs.
        features.b5g6r5_format = features.apple_gpu_family >= 1;

        physical_memory = NSProcessInfo.processInfo.physicalMemory;
    }
    Report();
}

Device::~Device() = default;

void Device::Report() const {
    using namespace Common::Literals;
    LOG_INFO(Render_Metal, "Device: {}", name);
    LOG_INFO(Render_Metal, "Apple GPU family: {}", features.apple_gpu_family);
    LOG_INFO(Render_Metal, "Metal 3: {}, argument buffers tier 2: {}, unified memory: {}",
             features.metal3, features.argument_buffers_tier2, features.unified_memory);
    LOG_INFO(Render_Metal, "Ray tracing: {}, BC texture compression: {}", features.raytracing,
             features.bc_texture_compression);
    LOG_INFO(Render_Metal, "Recommended working set: {:.2f} GiB, physical memory: {:.2f} GiB",
             static_cast<f64>(recommended_working_set_size) / f64{1_GiB},
             static_cast<f64>(physical_memory) / f64{1_GiB});
    if (!features.metal3) {
        LOG_WARNING(Render_Metal, "Device does not support Metal 3, the backend is untested here");
    }
}

} // namespace Metal
