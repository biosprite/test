// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include "common/common_types.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/tests/A64/testenv.h"
#include "dynarmic/tests/fuzz_util.h"
#include "dynarmic/tests/unicorn_emu/a64_unicorn.h"

// The dynarmic-versus-Unicorn runner behind fuzz_with_unicorn.cpp, shared with the table tests.
namespace UnicornFuzz {

// Everything one instance depends on; printed as an "a64-replay" line on a mismatch.
struct A64FuzzCase {
    A64Unicorn::RegisterArray regs{};
    A64Unicorn::VectorArray vecs{};
    u64 start_address = 0;
    u64 initial_sp = 0;
    std::vector<u32> instructions;
    u32 pstate = 0;
    u32 fpcr = 0;
};

Dynarmic::A64::UserConfig GetUserConfig(A64TestEnv& jit_env);
std::string ToReplayLine(const A64FuzzCase& c);
// Runs one case on both and returns whether they agree; a disagreement is reported as asked.
bool RunFuzzCase(Dynarmic::A64::Jit& jit, A64Unicorn& uni, A64TestEnv& jit_env, A64TestEnv& uni_env, const Dynarmic::A64::UserConfig& conf, const A64FuzzCase& c, size_t iteration, FuzzReport report);

}  // namespace UnicornFuzz
