// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>

#import <AppKit/AppKit.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_swapchain.h"

namespace Metal {

namespace {

CAMetalLayer* FindMetalLayer(CALayer* layer) {
    if (!layer) {
        return nil;
    }
    if ([layer isKindOfClass:[CAMetalLayer class]]) {
        return (CAMetalLayer*)layer;
    }
    // Qt 6 may wrap the Metal layer inside a container layer.
    for (CALayer* sublayer in layer.sublayers) {
        if ([sublayer isKindOfClass:[CAMetalLayer class]]) {
            return (CAMetalLayer*)sublayer;
        }
    }
    return nil;
}

/// The Qt frontend hands out a CAMetalLayer, the SDL frontend an NSView backed by one.
CAMetalLayer* ResolveMetalLayer(void* render_surface) {
    if (!render_surface) {
        return nil;
    }
    const id object = (__bridge id)render_surface;
    if ([object isKindOfClass:[CALayer class]]) {
        return FindMetalLayer((CALayer*)object);
    }
    if ([object isKindOfClass:[NSView class]]) {
        // Only reads the backing layer, the view hierarchy is not modified off the main thread.
        return FindMetalLayer(((NSView*)object).layer);
    }
    return nil;
}

} // Anonymous namespace

Swapchain::Swapchain(const Device& device,
                     const Core::Frontend::EmuWindow::WindowSystemInfo& window_info) {
    if (window_info.type != Core::Frontend::WindowSystemType::Cocoa) {
        throw std::runtime_error{"The Metal renderer requires a Cocoa window"};
    }
    @autoreleasepool {
        layer = ResolveMetalLayer(window_info.render_surface);
    }
    if (!layer) {
        throw std::runtime_error{"The render window does not provide a CAMetalLayer"};
    }

    @autoreleasepool {
        layer.device = device.GetMTLDevice();
        layer.pixelFormat = PixelFormat;
        layer.framebufferOnly = YES;
        layer.maximumDrawableCount = MaxDrawableCount;
        const Settings::VSyncMode mode = Settings::values.vsync_mode.GetValue();
        layer.displaySyncEnabled = mode != Settings::VSyncMode::Immediate;

        const CGSize size = layer.drawableSize;
        width = static_cast<u32>(size.width);
        height = static_cast<u32>(size.height);
    }
    LOG_INFO(Render_Metal, "Swapchain created, drawable size {}x{}, vsync {}", width, height,
             layer.displaySyncEnabled ? "on" : "off");
}

Swapchain::~Swapchain() = default;

void Swapchain::Resize(u32 new_width, u32 new_height) {
    if (new_width == 0 || new_height == 0) {
        return;
    }
    // The frontend may resize the layer on its own, so compare against the layer itself.
    const CGSize current = layer.drawableSize;
    if (static_cast<u32>(current.width) != new_width ||
        static_cast<u32>(current.height) != new_height) {
        layer.drawableSize = CGSizeMake(new_width, new_height);
    }
    width = new_width;
    height = new_height;
}

id<CAMetalDrawable> Swapchain::AcquireNextDrawable() {
    if (width == 0 || height == 0) {
        return nil;
    }
    // Blocks for at most one second when every drawable is in flight, returns nil on timeout.
    return [layer nextDrawable];
}

void Swapchain::Present(id<MTLCommandBuffer> command_buffer, id<CAMetalDrawable> drawable) {
    // With display sync on, the drawable waits for the next vblank on its own. A minimum
    // presentation duration on top of it held drawables longer than the frame cadence, which
    // starved nextDrawable and dropped frames.
    [command_buffer presentDrawable:drawable];
}

} // namespace Metal
