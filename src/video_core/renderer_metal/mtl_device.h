// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_device.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <string>

#import <Metal/Metal.h>

#include "common/common_types.h"

namespace Metal {

struct DeviceFeatures {
    /// Highest MTLGPUFamilyApple<N> supported, 0 when the GPU is not an Apple GPU.
    u32 apple_gpu_family{};
    bool metal3{};
    bool argument_buffers_tier2{};
    bool unified_memory{};
    bool raytracing{};
    bool bc_texture_compression{};
    bool b5g6r5_format{};
};

/// Owns the MTLDevice and the MTLCommandQueue shared by the whole renderer.
class Device {
public:
    /// @throws std::runtime_error when no Metal device is available.
    explicit Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] id<MTLDevice> GetMTLDevice() const noexcept {
        return device;
    }

    [[nodiscard]] id<MTLCommandQueue> GetCommandQueue() const noexcept {
        return queue;
    }

    [[nodiscard]] const std::string& GetName() const noexcept {
        return name;
    }

    [[nodiscard]] const DeviceFeatures& GetFeatures() const noexcept {
        return features;
    }

    [[nodiscard]] u64 GetRecommendedMaxWorkingSetSize() const noexcept {
        return recommended_working_set_size;
    }

    [[nodiscard]] u64 GetPhysicalMemory() const noexcept {
        return physical_memory;
    }

private:
    void Report() const;

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    std::string name;
    DeviceFeatures features;
    u64 recommended_working_set_size{};
    u64 physical_memory{};
};

} // namespace Metal
