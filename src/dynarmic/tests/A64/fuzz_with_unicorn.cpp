// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include "common/common_types.h"

#include "dynarmic/tests/fuzz_util.h"
#include "dynarmic/tests/rand_int.h"
#include "dynarmic/tests/unicorn_emu/a64_unicorn.h"
#include "dynarmic/tests/A64/testenv.h"
#include "dynarmic/tests/A64/unicorn_harness.h"
#include "dynarmic/tests/native/testenv.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/frontend/A64/decoder/a64.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/frontend/A64/translate/impl/impl.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt_passes.h"

// Must be declared last for all necessary operator<< to be declared prior to this.
#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Dynarmic;

static bool ShouldTestInst(u32 instruction, u64 pc, bool is_last_inst) {
    const A64::LocationDescriptor location{pc, {}};
    IR::Block block{location};
    bool should_continue = A64::TranslateSingleInstruction(block, location, instruction);
    if (!should_continue && !is_last_inst)
        return false;
    for (const auto& ir_inst : block.instructions) {
        switch (ir_inst.GetOpcode()) {
        case IR::Opcode::A64ExceptionRaised:
        case IR::Opcode::A64CallSupervisor:
        case IR::Opcode::A64DataCacheOperationRaised:
        case IR::Opcode::A64GetCNTPCT:
            return false;
        default:
            continue;
        }
    }
    return true;
}

// Encodings dynarmic decodes but does not implement: translating them aborts the process.
static bool IsUnimplementedInDynarmic(u32 instruction) {
    static const std::array unimplemented{
        InstructionGenerator{"0-101110010-----111-01----------"},  // FCADD, half-precision
    };
    return std::any_of(unimplemented.begin(), unimplemented.end(), [instruction](const auto& gen) { return gen.Match(instruction); });
}

static u32 GenRandomInst(u64 pc, bool is_last_inst) {
    static const struct InstructionGeneratorInfo {
        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;
    } instructions = [] {
        const std::vector<std::tuple<std::string, const char*>> list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A64/decoder/a64.inc"
#undef INST
        };

        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;

        // List of instructions not to test
        const std::vector<std::string> do_not_test{
            // Dynarmic and QEMU currently differ on how the exclusive monitor's address range works.
            "STXR",
            "STLXR",
            "STXP",
            "STLXP",
            "LDXR",
            "LDAXR",
            "LDXP",
            "LDAXP",
            // Behaviour differs from QEMU
            "MSR_reg",
            "MSR_imm",
            "MRS",
            // Does not need test
            "SVC",
            "BRK"
        };

        for (const auto& [fn, bitstring] : list) {
            if (fn == "UnallocatedEncoding") {
                continue;
            }
            if (std::find(do_not_test.begin(), do_not_test.end(), fn) != do_not_test.end()) {
                invalid.emplace_back(InstructionGenerator{bitstring});
                continue;
            }
            generators.emplace_back(InstructionGenerator{bitstring});
        }
        return InstructionGeneratorInfo{generators, invalid};
    }();

    while (true) {
        const size_t index = RandInt<size_t>(0, instructions.generators.size() - 1);
        const u32 inst = instructions.generators[index].Generate();

        if (std::any_of(instructions.invalid.begin(), instructions.invalid.end(), [inst](const auto& invalid) { return invalid.Match(inst); })) {
            continue;
        }
        if (IsUnimplementedInDynarmic(inst)) {
            continue;
        }
        if (ShouldTestInst(inst, pc, is_last_inst)) {
            return inst;
        }
    }
}

static u32 GenFloatInst(u64 pc, bool is_last_inst) {
    static const std::vector<InstructionGenerator> instruction_generators = [] {
        const std::vector<std::tuple<std::string, std::string, const char*>> list{
#define INST(fn, name, bitstring) {#fn, #name, bitstring},
#include "dynarmic/frontend/A64/decoder/a64.inc"
#undef INST
        };

        // List of instructions not to test
        const std::vector<std::string> do_not_test{};

        std::vector<InstructionGenerator> result;

        for (const auto& [fn, name, bitstring] : list) {
            (void)name;

            if (fn[0] != 'F') {
                continue;
            } else if (std::find(do_not_test.begin(), do_not_test.end(), fn) != do_not_test.end()) {
                continue;
            }
            result.emplace_back(InstructionGenerator{bitstring});
        }

        return result;
    }();

    while (true) {
        const size_t index = RandInt<size_t>(0, instruction_generators.size() - 1);
        const u32 instruction = instruction_generators[index].Generate();

        if (IsUnimplementedInDynarmic(instruction)) {
            continue;
        }
        if (ShouldTestInst(instruction, pc, is_last_inst)) {
            return instruction;
        }
    }
}

namespace UnicornFuzz {

Dynarmic::A64::UserConfig GetUserConfig(A64TestEnv& jit_env) {
    Dynarmic::A64::UserConfig jit_user_config{};
    jit_user_config.callbacks = &jit_env;
    jit_user_config.optimizations &= ~OptimizationFlag::FastDispatch;
    // The below corresponds to the settings for qemu's aarch64_max_initfn
    jit_user_config.dczid_el0 = 7;
    jit_user_config.ctr_el0 = 0x80038003;
    return jit_user_config;
}


std::string ToReplayLine(const A64FuzzCase& c) {
    std::string line = fmt::format("a64-replay pc={:x} sp={:x} nzcv={:x} fpcr={:x} insts=", c.start_address, c.initial_sp, c.pstate, c.fpcr);
    for (size_t i = 0; i < c.instructions.size(); ++i)
        line += fmt::format("{}{:08x}", i ? "," : "", c.instructions[i]);
    line += " x=";
    for (size_t i = 0; i < c.regs.size(); ++i)
        line += fmt::format("{}{:x}", i ? "," : "", c.regs[i]);
    line += " v=";
    for (size_t i = 0; i < c.vecs.size(); ++i)
        line += fmt::format("{}{:016x}{:016x}", i ? "," : "", c.vecs[i][1], c.vecs[i][0]);
    return line;
}

std::optional<A64FuzzCase> FromReplayLine(const std::string& line) {
    std::istringstream in{line};
    std::string token;
    if (!(in >> token) || token != "a64-replay")
        return std::nullopt;
    const auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream parts{s};
        for (std::string part; std::getline(parts, part, ',');)
            out.push_back(part);
        return out;
    };
    const auto hex = [](const std::string& s) { return std::strtoull(s.c_str(), nullptr, 16); };
    A64FuzzCase c;
    while (in >> token) {
        const size_t eq = token.find('=');
        if (eq == std::string::npos)
            return std::nullopt;
        const std::string key = token.substr(0, eq);
        const std::string value = token.substr(eq + 1);
        if (key == "pc") {
            c.start_address = hex(value);
        } else if (key == "sp") {
            c.initial_sp = hex(value);
        } else if (key == "nzcv") {
            c.pstate = u32(hex(value));
        } else if (key == "fpcr") {
            c.fpcr = u32(hex(value));
        } else if (key == "insts") {
            for (const auto& s : split(value))
                c.instructions.push_back(u32(hex(s)));
        } else if (key == "x") {
            const auto xs = split(value);
            for (size_t i = 0; i < xs.size() && i < c.regs.size(); ++i)
                c.regs[i] = hex(xs[i]);
        } else if (key == "v") {
            const auto vs = split(value);
            for (size_t i = 0; i < vs.size() && i < c.vecs.size(); ++i) {
                const std::string padded = std::string(32 - std::min<size_t>(32, vs[i].size()), '0') + vs[i];
                c.vecs[i] = Vector{hex(padded.substr(16, 16)), hex(padded.substr(0, 16))};
            }
        }
    }
    if (c.instructions.empty())
        return std::nullopt;
    return c;
}

std::string_view A64Name(u32 instruction) {
    const auto name = A64::GetName<A64::TranslatorVisitor>(instruction);
    return name ? *name : std::string_view{"<unallocated>"};
}

void DumpMismatch(Dynarmic::A64::Jit& jit, A64Unicorn& uni, A64TestEnv& jit_env, A64TestEnv& uni_env, const Dynarmic::A64::UserConfig& conf, const A64FuzzCase& c, size_t iteration, const std::string& diff, bool full) {
    fmt::print("=== fuzz mismatch: test=\"{}\" seed={} iteration={} diff={}\n", Catch::getResultCapture().getCurrentTestName(), Catch::getSeed(), iteration, diff);
    for (size_t i = 0; i < c.instructions.size(); ++i)
        fmt::print("  inst {:016x}: {:08x} {}\n", c.start_address + i * 4, c.instructions[i], A64Name(c.instructions[i]));
    if (c.instructions.size() == 1) {
        // Initial values of the registers the d, n, a and m fields can name.
        const u32 inst = c.instructions[0];
        const std::array<std::pair<char, u32>, 4> fields{{{'d', inst & 31}, {'n', (inst >> 5) & 31}, {'a', (inst >> 10) & 31}, {'m', (inst >> 16) & 31}}};
        for (const auto& [field, index] : fields)
            fmt::print("  {}: x{} {:016x}  v{} {:016x}{:016x}\n", field, index, index < 31 ? c.regs[index] : 0, index, c.vecs[index][1], c.vecs[index][0]);
    }
    const FP::FPCR fpcr{c.fpcr};
    fmt::print("  nzcv {:08x} fpcr {:08x} (AHP {} DN {} FZ {} RMode {} FZ16 {})\n", c.pstate, c.fpcr, fpcr.AHP(), fpcr.DN(), fpcr.FZ(), static_cast<size_t>(fpcr.RMode()), fpcr.FZ16());
    fmt::print("  {:>5} {:>32} {:>32} {:>32}\n", "", "initial", "unicorn", "dynarmic");
    const auto uni_regs = uni.GetRegisters();
    const auto jit_regs = jit.GetRegisters();
    for (size_t i = 0; i < c.regs.size(); ++i) {
        if (uni_regs[i] != jit_regs[i])
            fmt::print("  {:>5} {:>32x} {:>32x} {:>32x} *\n", A64::RegToString(static_cast<A64::Reg>(i)), c.regs[i], uni_regs[i], jit_regs[i]);
    }
    const auto uni_vecs = uni.GetVectors();
    const auto jit_vecs = jit.GetVectors();
    for (size_t i = 0; i < c.vecs.size(); ++i) {
        if (uni_vecs[i] != jit_vecs[i])
            fmt::print("  {:>5} {:016x}{:016x} {:016x}{:016x} {:016x}{:016x} *\n", A64::VecToString(static_cast<A64::Vec>(i)), c.vecs[i][1], c.vecs[i][0], uni_vecs[i][1], uni_vecs[i][0], jit_vecs[i][1], jit_vecs[i][0]);
    }
    fmt::print("  sp {:x} -> {:x} {:x}\n", c.initial_sp, uni.GetSP(), jit.GetSP());
    fmt::print("  pc {:x} -> {:x} {:x}\n", c.start_address, uni.GetPC(), jit.GetPC());
    fmt::print("  nzcv -> {:08x} {:08x}, fpsr -> {:08x} {:08x}\n", uni.GetPstate(), jit.GetPstate(), uni.GetFpsr(), jit.GetFpsr());
    std::map<u64, std::pair<int, int>> memory;
    for (const auto& [addr, value] : uni_env.modified_memory)
        memory[addr].first = value + 1;
    for (const auto& [addr, value] : jit_env.modified_memory)
        memory[addr].second = value + 1;
    for (const auto& [addr, values] : memory) {
        if (values.first != values.second)
            fmt::print("  mem {:016x}: {:>2} {:>2} *\n", addr, values.first ? fmt::format("{:02x}", values.first - 1) : "--", values.second ? fmt::format("{:02x}", values.second - 1) : "--");
    }
    for (const auto& interrupt : uni_env.interrupts)
        fmt::print("  unicorn: {}\n", interrupt);
    for (const auto& exception : jit_env.exceptions)
        fmt::print("  dynarmic: {}\n", exception);
    if (full) {
        const A64::LocationDescriptor location{c.start_address, fpcr};
        IR::Block ir_block{location};
        A64::Translate(ir_block, location, conf);
        fmt::print("IR:\n{}\n", IR::DumpBlock(ir_block));
        Optimization::Optimize(ir_block, conf, {});
        fmt::print("Optimized IR:\n{}\n", IR::DumpBlock(ir_block));
        if (std::getenv("DYNARMIC_FUZZ_DUMP_X64"))
            fmt::print("x86_64:\n{}\n", jit.Disassemble());
    }
    fmt::print("{}\n", ToReplayLine(c));
    std::fflush(stdout);
}

// Runs one case on both and returns whether they agree; a disagreement is reported as asked.
bool RunFuzzCase(Dynarmic::A64::Jit& jit, A64Unicorn& uni, A64TestEnv& jit_env, A64TestEnv& uni_env, const Dynarmic::A64::UserConfig& conf, const A64FuzzCase& c, size_t iteration, FuzzReport report) {
    // DYNARMIC_FUZZ_TRACE=1 prints each case first, so a case that aborts the process is on record.
    static const bool trace = std::getenv("DYNARMIC_FUZZ_TRACE") != nullptr;
    if (trace) {
        fmt::print("trace {}\n", ToReplayLine(c));
        std::fflush(stdout);
    }
    jit_env.code_mem = c.instructions;
    uni_env.code_mem = c.instructions;
    jit_env.code_mem.emplace_back(0x14000000);  // B .
    uni_env.code_mem.emplace_back(0x14000000);  // B .
    jit_env.code_mem_start_address = c.start_address;
    uni_env.code_mem_start_address = c.start_address;
    jit_env.modified_memory.clear();
    uni_env.modified_memory.clear();
    jit_env.interrupts.clear();
    uni_env.interrupts.clear();
    jit_env.record_exceptions = true;
    jit_env.exceptions.clear();
    jit_env.code_mem_modified_by_guest = false;
    uni_env.code_mem_modified_by_guest = false;

    jit.SetRegisters(c.regs);
    jit.SetVectors(c.vecs);
    jit.SetPC(c.start_address);
    jit.SetSP(c.initial_sp);
    jit.SetFpcr(c.fpcr);
    jit.SetFpsr(0);
    jit.SetPstate(c.pstate);
    jit.ClearCache();
    uni.SetRegisters(c.regs);
    uni.SetVectors(c.vecs);
    uni.SetPC(c.start_address);
    uni.SetSP(c.initial_sp);
    uni.SetFpcr(c.fpcr);
    uni.SetFpsr(0);
    uni.SetPstate(c.pstate);
    uni.ClearPageCache();

    jit_env.ticks_left = c.instructions.size();
    CheckedRun([&]() { jit.Run(); });

    uni_env.ticks_left = c.instructions.size() * 4;
    uni.Run();

    if (uni_env.code_mem_modified_by_guest != jit_env.code_mem_modified_by_guest) {
        if (report != FuzzReport::Quiet)
            DumpMismatch(jit, uni, jit_env, uni_env, conf, c, iteration, "code_mem_modified", report == FuzzReport::Full);
        return false;
    }
    if (uni_env.code_mem_modified_by_guest)
        return true;

    std::string diff;
    const auto note = [&diff](bool equal, const char* what) {
        if (!equal)
            diff += diff.empty() ? what : fmt::format(",{}", what);
    };
    // Unicorn 2 stops on the trailing B . like the JIT does; Unicorn 1 stopped one short.
    note(uni.GetPC() == jit.GetPC(), "pc");
    note(uni.GetRegisters() == jit.GetRegisters(), "regs");
    note(uni.GetVectors() == jit.GetVectors(), "vecs");
    note(uni.GetSP() == jit.GetSP(), "sp");
    note((uni.GetPstate() & 0xF0000000) == (jit.GetPstate() & 0xF0000000), "nzcv");
    note(uni_env.modified_memory == jit_env.modified_memory, "mem");
    note(uni_env.interrupts.empty() == jit_env.exceptions.empty(), "interrupt");
    note(FP::FPSR{uni.GetFpsr()}.QC() == FP::FPSR{jit.GetFpsr()}.QC(), "qc");
    if (diff.empty())
        return true;

    if (report != FuzzReport::Quiet)
        DumpMismatch(jit, uni, jit_env, uni_env, conf, c, iteration, diff, report == FuzzReport::Full);
    return false;
}

}  // namespace UnicornFuzz

using namespace UnicornFuzz;

static bool RunTestInstance(Dynarmic::A64::Jit& jit, A64Unicorn& uni, A64TestEnv& jit_env, A64TestEnv& uni_env, Dynarmic::A64::UserConfig& conf, const A64Unicorn::RegisterArray& regs, const A64Unicorn::VectorArray& vecs, const size_t instructions_start, const std::vector<u32>& instructions, const u32 pstate, const u32 fpcr, size_t iteration) {
    A64FuzzCase c;
    c.regs = regs;
    c.vecs = vecs;
    c.start_address = instructions_start;
    c.initial_sp = RandInt<u64>(0x30'0000'0000, 0x40'0000'0000) * 4;
    c.instructions = instructions;
    c.pstate = pstate;
    c.fpcr = fpcr;
    const bool ok = RunFuzzCase(jit, uni, jit_env, uni_env, conf, c, iteration, FuzzKeepGoing() ? FuzzReport::Brief : FuzzReport::Full);
    if (!ok && !FuzzKeepGoing())
        FAIL("dynarmic and Unicorn disagree, see the fuzz mismatch above");
    return ok;
}

TEST_CASE("A64: Single random instruction", "[a64][unicorn]") {
    A64TestEnv jit_env{};
    A64TestEnv uni_env{};

    Dynarmic::A64::UserConfig conf{GetUserConfig(jit_env)};
    Dynarmic::A64::Jit jit{conf};
    A64Unicorn uni{uni_env};

    A64Unicorn::RegisterArray regs;
    A64Unicorn::VectorArray vecs;
    std::vector<u32> instructions(1);

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u64>(0, ~u64(0)); });
        std::generate(vecs.begin(), vecs.end(), RandomVector);

        instructions[0] = GenRandomInst(0, true);

        const u64 start_address = RandInt<u64>(0, 0x10'0000'0000) * 4;
        const u32 pstate = RandInt<u32>(0, 0xF) << 28;
        const u32 fpcr = RandomFpcr();

        INFO("Instruction: 0x" << std::hex << instructions[0]);

        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, conf, regs, vecs, start_address, instructions, pstate, fpcr, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A64: Floating point instructions", "[a64][unicorn]") {
    A64TestEnv jit_env{};
    A64TestEnv uni_env{};

    Dynarmic::A64::UserConfig conf{GetUserConfig(jit_env)};
    Dynarmic::A64::Jit jit{conf};
    A64Unicorn uni{uni_env};

    static constexpr std::array<u64, 80> float_numbers{
        0x00000000,  // positive zero
        0x00000001,  // smallest positive denormal
        0x00000076,  //
        0x00002b94,  //
        0x00636d24,  //
        0x007fffff,  // largest positive denormal
        0x00800000,  // smallest positive normalised real
        0x00800002,  //
        0x01398437,  //
        0x0ba98d27,  //
        0x0ba98d7a,  //
        0x751f853a,  //
        0x7f7ffff0,  //
        0x7f7fffff,  // largest positive normalised real
        0x7f800000,  // positive infinity
        0x7f800001,  // first positive SNaN
        0x7f984a37,  //
        0x7fbfffff,  // last positive SNaN
        0x7fc00000,  // first positive QNaN
        0x7fd9ba98,  //
        0x7fffffff,  // last positive QNaN
        0x80000000,  // negative zero
        0x80000001,  // smallest negative denormal
        0x80000076,  //
        0x80002b94,  //
        0x80636d24,  //
        0x807fffff,  // largest negative denormal
        0x80800000,  // smallest negative normalised real
        0x80800002,  //
        0x81398437,  //
        0x8ba98d27,  //
        0x8ba98d7a,  //
        0xf51f853a,  //
        0xff7ffff0,  //
        0xff7fffff,  // largest negative normalised real
        0xff800000,  // negative infinity
        0xff800001,  // first negative SNaN
        0xff984a37,  //
        0xffbfffff,  // last negative SNaN
        0xffc00000,  // first negative QNaN
        0xffd9ba98,  //
        0xffffffff,  // last negative QNaN
        // some random numbers follow
        0x4f3495cb,
        0xe73a5134,
        0x7c994e9e,
        0x6164bd6c,
        0x09503366,
        0xbf5a97c9,
        0xe6ff1a14,
        0x77f31e2f,
        0xaab4d7d8,
        0x0966320b,
        0xb26bddee,
        0xb5c8e5d3,
        0x317285d3,
        0x3c9623b1,
        0x51fd2c7c,
        0x7b906a6c,
        0x3f800000,
        0x3dcccccd,
        0x3f000000,
        0x42280000,
        0x3eaaaaab,
        0xc1200000,
        0xbf800000,
        0xbf8147ae,
        0x3f8147ae,
        0x415df525,
        0xc79b271e,
        0x460e8c84,
        // some 64-bit-float upper-halves
        0x7ff00000,  // +SNaN / +Inf
        0x7ff0abcd,  // +SNaN
        0x7ff80000,  // +QNaN
        0x7ff81234,  // +QNaN
        0xfff00000,  // -SNaN / -Inf
        0xfff05678,  // -SNaN
        0xfff80000,  // -QNaN
        0xfff809ef,  // -QNaN
        0x3ff00000,  // Number near +1.0
        0xbff00000,  // Number near -1.0
    };

    const auto gen_float = [&] {
        if (RandInt<size_t>(0, 1) == 0) {
            return RandInt<u64>(0, 0xffffffff);
        }
        return float_numbers[RandInt<size_t>(0, float_numbers.size() - 1)];
    };

    const auto gen_vector = [&] {
        u64 upper = (gen_float() << 32) | gen_float();
        u64 lower = (gen_float() << 32) | gen_float();
        return Vector{lower, upper};
    };

    A64Unicorn::RegisterArray regs;
    A64Unicorn::VectorArray vecs;
    std::vector<u32> instructions(1);

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), gen_float);
        std::generate(vecs.begin(), vecs.end(), gen_vector);

        instructions[0] = GenFloatInst(0, true);

        const u64 start_address = RandInt<u64>(0, 0x10'0000'0000) * 4;
        const u32 pstate = RandInt<u32>(0, 0xF) << 28;
        const u32 fpcr = RandomFpcr();

        INFO("Instruction: 0x" << std::hex << instructions[0]);

        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, conf, regs, vecs, start_address, instructions, pstate, fpcr, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A64: Small random block", "[a64][unicorn]") {
    A64TestEnv jit_env{};
    A64TestEnv uni_env{};

    Dynarmic::A64::UserConfig conf{GetUserConfig(jit_env)};
    Dynarmic::A64::Jit jit{conf};
    A64Unicorn uni{uni_env};

    A64Unicorn::RegisterArray regs;
    A64Unicorn::VectorArray vecs;
    std::vector<u32> instructions(5);

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u64>(0, ~u64(0)); });
        std::generate(vecs.begin(), vecs.end(), RandomVector);

        instructions[0] = GenRandomInst(0, false);
        instructions[1] = GenRandomInst(4, false);
        instructions[2] = GenRandomInst(8, false);
        instructions[3] = GenRandomInst(12, false);
        instructions[4] = GenRandomInst(16, true);

        const u64 start_address = RandInt<u64>(0, 0x10'0000'0000) * 4;
        const u32 pstate = RandInt<u32>(0, 0xF) << 28;
        const u32 fpcr = RandomFpcr();

        INFO("Instruction 1: 0x" << std::hex << instructions[0]);
        INFO("Instruction 2: 0x" << std::hex << instructions[1]);
        INFO("Instruction 3: 0x" << std::hex << instructions[2]);
        INFO("Instruction 4: 0x" << std::hex << instructions[3]);
        INFO("Instruction 5: 0x" << std::hex << instructions[4]);

        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, conf, regs, vecs, start_address, instructions, pstate, fpcr, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A64: Large random block", "[a64][unicorn]") {
    A64TestEnv jit_env{};
    A64TestEnv uni_env{};

    Dynarmic::A64::UserConfig conf{GetUserConfig(jit_env)};
    Dynarmic::A64::Jit jit{conf};
    A64Unicorn uni{uni_env};

    A64Unicorn::RegisterArray regs;
    A64Unicorn::VectorArray vecs;

    constexpr size_t instruction_count = 100;
    std::vector<u32> instructions(instruction_count);

    const size_t iterations = FuzzIterations(500);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u64>(0, ~u64(0)); });
        std::generate(vecs.begin(), vecs.end(), RandomVector);

        for (size_t j = 0; j < instruction_count; ++j) {
            instructions[j] = GenRandomInst(j * 4, j == instruction_count - 1);
        }

        const u64 start_address = RandInt<u64>(0, 0x10'0000'0000) * 4;
        const u32 pstate = RandInt<u32>(0, 0xF) << 28;
        const u32 fpcr = RandomFpcr();

        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, conf, regs, vecs, start_address, instructions, pstate, fpcr, iteration);
    }
    REQUIRE(mismatches == 0);
}

// DYNARMIC_FUZZ_REPLAY holds one "a64-replay ..." line, or names a file of them.
// DYNARMIC_FUZZ_MINIMIZE=1 also turns every instruction a block mismatch does not need into NOP.
TEST_CASE("A64: Replay fuzz cases", "[.][a64][unicorn]") {
    const char* replay = std::getenv("DYNARMIC_FUZZ_REPLAY");
    if (!replay)
        SKIP("set DYNARMIC_FUZZ_REPLAY to a replay line or a file of them");
    std::vector<std::string> lines;
    if (std::string{replay}.starts_with("a64-replay")) {
        lines.emplace_back(replay);
    } else {
        std::ifstream file{replay};
        for (std::string line; std::getline(file, line);) {
            if (line.starts_with("a64-replay"))
                lines.push_back(line);
        }
    }
    REQUIRE(!lines.empty());

    A64TestEnv jit_env{};
    A64TestEnv uni_env{};
    Dynarmic::A64::UserConfig conf{GetUserConfig(jit_env)};
    Dynarmic::A64::Jit jit{conf};
    A64Unicorn uni{uni_env};

    size_t mismatches = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto c = FromReplayLine(lines[i]);
        REQUIRE(c.has_value());
        if (RunFuzzCase(jit, uni, jit_env, uni_env, conf, *c, i, FuzzReport::Full))
            continue;
        ++mismatches;
        if (!std::getenv("DYNARMIC_FUZZ_MINIMIZE") || c->instructions.size() < 2)
            continue;
        A64FuzzCase minimized = *c;
        for (auto& instruction : minimized.instructions) {
            const u32 original = std::exchange(instruction, 0xd503201f);  // NOP
            if (RunFuzzCase(jit, uni, jit_env, uni_env, conf, minimized, i, FuzzReport::Quiet))
                instruction = original;
        }
        fmt::print("minimized:\n");
        RunFuzzCase(jit, uni, jit_env, uni_env, conf, minimized, i, FuzzReport::Full);
    }
    REQUIRE(mismatches == 0);
}
