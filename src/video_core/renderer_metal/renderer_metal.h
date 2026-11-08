// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"

namespace Tegra {
class GPU;
}

namespace Metal {

class RendererMetal final : public VideoCore::RendererBase {
public:
    /// @throws std::runtime_error when Metal or the window surface cannot be initialized.
    explicit RendererMetal(Core::Frontend::EmuWindow& emu_window,
                           Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu,
                           std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererMetal() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override {
        return "Apple";
    }

private:
    /// Objective-C++ state (device, caches, rasterizer, swapchain), kept out of this header so
    /// that it can be included from plain C++ translation units.
    struct Impl;

    Tegra::GPU& gpu;
    std::unique_ptr<Impl> impl;
    /// Owned by impl.
    VideoCore::RasterizerInterface* rasterizer{};
};

} // namespace Metal
