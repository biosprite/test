// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include "shader_recompiler/frontend/ir/condition.h"

namespace Shader::IR {

std::string NameOf(Condition condition) {
    std::string ret;
    if (condition.GetFlowTest() != FlowTest::T) {
        ret = std::to_string(std::to_underlying(condition.GetFlowTest()));
    }
    const auto [pred, negated]{condition.GetPred()};
    if (!ret.empty()) {
        ret += '&';
    }
    if (negated) {
        ret += '!';
    }
    ret += std::to_string(std::to_underlying(pred));
    return ret;
}

} // namespace Shader::IR
