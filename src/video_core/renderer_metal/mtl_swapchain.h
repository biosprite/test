// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_swapchain.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "common/common_types.h"
#include "core/frontend/emu_window.h"

namespace Metal {

class Device;

/// Wraps the CAMetalLayer provided by the frontend window.
class Swapchain {
public:
    static constexpr MTLPixelFormat PixelFormat = MTLPixelFormatBGRA8Unorm;
    /// Drawables the layer hands out before AcquireNextDrawable blocks.
    static constexpr NSUInteger MaxDrawableCount = 3;

    /// @throws std::runtime_error when the window does not expose a CAMetalLayer.
    explicit Swapchain(const Device& device,
                       const Core::Frontend::EmuWindow::WindowSystemInfo& window_info);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// Updates the drawable size of the layer, in physical pixels.
    void Resize(u32 width, u32 height);

    /// Returns the next drawable, or nil when none is available. Must be called inside an
    /// autorelease pool, the returned drawable is autoreleased.
    [[nodiscard]] id<CAMetalDrawable> AcquireNextDrawable();

    /// Schedules the presentation of the drawable once the command buffer completes.
    void Present(id<MTLCommandBuffer> command_buffer, id<CAMetalDrawable> drawable);

    [[nodiscard]] u32 GetWidth() const noexcept {
        return width;
    }

    [[nodiscard]] u32 GetHeight() const noexcept {
        return height;
    }

private:
    CAMetalLayer* layer;
    u32 width{};
    u32 height{};
};

} // namespace Metal
