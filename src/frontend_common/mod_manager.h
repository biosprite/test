// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <vector>
#include "common/common_types.h"

namespace FrontendCommon {

enum ModInstallResult {
    Cancelled,
    Failed,
    Success,
};

std::vector<std::filesystem::path> GetModFolder(const std::string& root);

// If name is empty, the mod is named after its folder.
ModInstallResult InstallMod(const std::filesystem::path &path, const u64 program_id, const bool copy = true,
                            const std::string& name = {});
}
