// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Extra juicy formatter specializations.
#pragma once

#include <format>
#include <utility>

template <typename E>
    requires std::is_enum_v<E>
struct std::formatter<E> : std::formatter<std::string> {
    constexpr auto format(const E& e, auto& ctx) const {
        return std::formatter<std::string>::format(std::to_string(std::to_underlying(e)), ctx);
    }
};
