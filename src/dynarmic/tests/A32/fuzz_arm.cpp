// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include "dynarmic/mcl/bit.hpp"
#include "common/common_types.h"

#include "dynarmic/tests/fuzz_util.h"
#include "dynarmic/tests/rand_int.h"
#include "dynarmic/tests/unicorn_emu/a32_unicorn.h"
#include "dynarmic/tests/A32/testenv.h"
#include "dynarmic/tests/native/testenv.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include "dynarmic/frontend/A32/ITState.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/frontend/A32/decoder/arm.h"
#include "dynarmic/frontend/A32/decoder/asimd.h"
#include "dynarmic/frontend/A32/decoder/thumb16.h"
#include "dynarmic/frontend/A32/decoder/thumb32.h"
#include "dynarmic/frontend/A32/decoder/vfp.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/frontend/A32/translate/impl/a32_translate_impl.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/opcodes.h"

// Must be declared last for all necessary operator<< to be declared prior to this.
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace {
using namespace Dynarmic;

template<typename Fn>
bool AnyLocationDescriptorForTerminalHas(IR::Term::Terminal terminal, Fn fn) {
    if (auto const e = std::get_if<IR::Term::LeafTerminal>(&terminal)) {
        if (auto const x = std::get_if<IR::Term::LinkBlock>(e))
            return fn(x->next);
        if (auto const x = std::get_if<IR::Term::LinkBlockFast>(e))
            return fn(x->next);
    }
    if (auto const x = std::get_if<IR::Term::If>(&terminal))
        return AnyLocationDescriptorForTerminalHas(x->then_, fn) || AnyLocationDescriptorForTerminalHas(x->else_, fn);
    if (auto const x = std::get_if<IR::Term::CheckBit>(&terminal))
        return AnyLocationDescriptorForTerminalHas(x->then_, fn) || AnyLocationDescriptorForTerminalHas(x->else_, fn);
    if (auto const x = std::get_if<IR::Term::CheckHalt>(&terminal))
        return AnyLocationDescriptorForTerminalHas(x->else_, fn);
    return false;
}

bool ShouldTestInst(u32 instruction, u32 pc, bool is_thumb, bool is_last_inst, A32::ITState it_state = {}) {
    const A32::LocationDescriptor location = A32::LocationDescriptor{pc, {}, {}}.SetTFlag(is_thumb).SetIT(it_state);
    IR::Block block{location};
    const bool should_continue = A32::TranslateSingleInstruction(block, location, instruction);

    if (!should_continue && !is_last_inst) {
        return false;
    }

    if (AnyLocationDescriptorForTerminalHas(block.GetTerminal(), [&](IR::LocationDescriptor ld) { return A32::LocationDescriptor{ld}.PC() <= pc; })) {
        return false;
    }

    for (const auto& ir_inst : block.instructions) {
        switch (ir_inst.GetOpcode()) {
        case IR::Opcode::A32ExceptionRaised:
        case IR::Opcode::A32CallSupervisor:
        case IR::Opcode::A32CoprocInternalOperation:
        case IR::Opcode::A32CoprocSendOneWord:
        case IR::Opcode::A32CoprocSendTwoWords:
        case IR::Opcode::A32CoprocGetOneWord:
        case IR::Opcode::A32CoprocGetTwoWords:
        case IR::Opcode::A32CoprocLoadWords:
        case IR::Opcode::A32CoprocStoreWords:
            return false;
        // Currently unimplemented in Unicorn
        case IR::Opcode::FPVectorRecipEstimate16:
        case IR::Opcode::FPVectorRSqrtEstimate16:
        case IR::Opcode::VectorPolynomialMultiplyLong64:
            return false;
        default:
            continue;
        }
    }

    return true;
}

u32 GenRandomArmInst(u32 pc, bool is_last_inst) {
    static const struct InstructionGeneratorInfo {
        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;
    } instructions = [] {
        const std::vector<std::tuple<std::string, const char*>> list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/arm.inc"
#include "dynarmic/frontend/A32/decoder/asimd.inc"
#include "dynarmic/frontend/A32/decoder/vfp.inc"
#undef INST
        };

        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;

        // List of instructions not to test
        static constexpr std::array do_not_test{
            // Translating load/stores
            "arm_LDRBT", "arm_LDRBT", "arm_LDRHT", "arm_LDRHT", "arm_LDRSBT", "arm_LDRSBT", "arm_LDRSHT", "arm_LDRSHT", "arm_LDRT", "arm_LDRT",
            "arm_STRBT", "arm_STRBT", "arm_STRHT", "arm_STRHT", "arm_STRT", "arm_STRT",
            // Exclusive load/stores
            "arm_LDREXB", "arm_LDREXD", "arm_LDREXH", "arm_LDREX", "arm_LDAEXB", "arm_LDAEXD", "arm_LDAEXH", "arm_LDAEX",
            "arm_STREXB", "arm_STREXD", "arm_STREXH", "arm_STREX", "arm_STLEXB", "arm_STLEXD", "arm_STLEXH", "arm_STLEX",
            "arm_SWP", "arm_SWPB",
            // Elevated load/store multiple instructions.
            "arm_LDM_eret", "arm_LDM_usr",
            "arm_STM_usr",
            // Hint instructions
            "arm_NOP", "arm_PLD_imm", "arm_PLD_reg", "arm_SEV",
            "arm_WFE", "arm_WFI", "arm_YIELD",
            // E, T, J
            "arm_BLX_reg", "arm_BLX_imm", "arm_BXJ", "arm_SETEND",
            // Coprocessor
            "arm_CDP", "arm_LDC", "arm_MCR", "arm_MCRR", "arm_MRC", "arm_MRRC", "arm_STC",
            // System
            "arm_CPS", "arm_RFE", "arm_SRS",
            // Undefined
            "arm_UDF",
            // FPSCR is inaccurate
            "vfp_VMRS",
            // Incorrect Unicorn implementations
            "asimd_VRECPS",   // Unicorn does not fuse the multiply and subtraction, resulting in being off by 1ULP.
            "asimd_VRSQRTS",  // Unicorn does not fuse the multiply and subtraction, resulting in being off by 1ULP.
        };

        for (const auto& [fn, bitstring] : list) {
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

        if ((instructions.generators[index].Mask() & 0xF0000000) == 0 && (inst & 0xF0000000) == 0xF0000000) {
            continue;
        }
        // Another entry's pattern can produce an excluded encoding, as A64's generator knows.
        if (std::any_of(instructions.invalid.begin(), instructions.invalid.end(), [inst](const auto& invalid) { return invalid.Match(inst); })) {
            continue;
        }

        if (ShouldTestInst(inst, pc, false, is_last_inst)) {
            return inst;
        }
    }
}

std::vector<u16> GenRandomThumbInst(u32 pc, bool is_last_inst, A32::ITState it_state = {}) {
    static const struct InstructionGeneratorInfo {
        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;
    } instructions = [] {
        const std::vector<std::tuple<std::string, const char*>> list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/thumb16.inc"
#include "dynarmic/frontend/A32/decoder/thumb32.inc"
#undef INST
        };

        const std::vector<std::tuple<std::string, const char*>> vfp_list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/vfp.inc"
#undef INST
        };

        const std::vector<std::tuple<std::string, const char*>> asimd_list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/asimd.inc"
#undef INST
        };

        std::vector<InstructionGenerator> generators;
        std::vector<InstructionGenerator> invalid;

        // List of instructions not to test
        static constexpr std::array do_not_test{
            "thumb16_BKPT",
            "thumb16_IT",
            "thumb16_SETEND",

            // Exclusive load/stores
            "thumb32_LDREX",
            "thumb32_LDREXB",
            "thumb32_LDREXD",
            "thumb32_LDREXH",
            "thumb32_STREX",
            "thumb32_STREXB",
            "thumb32_STREXD",
            "thumb32_STREXH",

            // FPSCR is inaccurate
            "vfp_VMRS",

            // User-mode MRS leaves bits other than NZCVQ and GE UNKNOWN; Unicorn clears T, dynarmic keeps it.
            "thumb32_MRS_reg",
            // Unicorn is incorrect?
            "thumb32_MSR_reg",

            // Unicorn has incorrect implementation
            "asimd_VRECPS",   // Unicorn does not fuse the multiply and subtraction, resulting in being off by 1ULP.
            "asimd_VRSQRTS",  // Unicorn does not fuse the multiply and subtraction, resulting in being off by 1ULP.

            // E, T, J, as for arm_BXJ
            "thumb32_BXJ",
            // Random register targets can land back inside the test code, which both then run as
            // different garbage; "Fuzz Thumb instructions set 2" covers these with sane targets.
            "thumb16_BX",
            "thumb16_BLX_reg",

            // Coprocessor
            "thumb32_CDP",
            "thumb32_LDC",
            "thumb32_MCR",
            "thumb32_MCRR",
            "thumb32_MRC",
            "thumb32_MRRC",
            "thumb32_STC",
        };

        for (const auto& [fn, bitstring] : list) {
            if (std::find(do_not_test.begin(), do_not_test.end(), fn) != do_not_test.end()) {
                invalid.emplace_back(InstructionGenerator{bitstring});
                continue;
            }
            generators.emplace_back(InstructionGenerator{bitstring});
        }
        for (const auto& [fn, bs] : vfp_list) {
            std::string bitstring = bs;
            if (bitstring.substr(0, 4) == "cccc" || bitstring.substr(0, 4) == "----") {
                bitstring.replace(0, 4, "1110");
            }
            if (std::find(do_not_test.begin(), do_not_test.end(), fn) != do_not_test.end()) {
                invalid.emplace_back(InstructionGenerator{bitstring.c_str()});
                continue;
            }
            generators.emplace_back(InstructionGenerator{bitstring.c_str()});
        }
        for (const auto& [fn, bs] : asimd_list) {
            std::string bitstring = bs;
            if (bitstring.substr(0, 7) == "1111001") {
                const char U = bitstring[7];
                bitstring.replace(0, 8, "111-1111");
                bitstring[3] = U;
            } else if (bitstring.substr(0, 8) == "11110100") {
                bitstring.replace(0, 8, "11111001");
            } else {
                UNREACHABLE(); // "Unhandled ASIMD instruction: {} {}", fn, bs);
            }
            if (std::find(do_not_test.begin(), do_not_test.end(), fn) != do_not_test.end()) {
                invalid.emplace_back(InstructionGenerator{bitstring.c_str()});
                continue;
            }
            generators.emplace_back(InstructionGenerator{bitstring.c_str()});
        }
        return InstructionGeneratorInfo{generators, invalid};
    }();

    while (true) {
        const size_t index = RandInt<size_t>(0, instructions.generators.size() - 1);
        const u32 inst = instructions.generators[index].Generate();
        const bool is_four_bytes = (inst >> 16) != 0;
        if (std::any_of(instructions.invalid.begin(), instructions.invalid.end(), [inst](const auto& invalid) { return invalid.Match(inst); })) {
            continue;
        }

        if (ShouldTestInst(is_four_bytes ? mcl::bit::swap_halves_32(inst) : inst, pc, true, is_last_inst, it_state)) {
            if (is_four_bytes)
                return {static_cast<u16>(inst >> 16), static_cast<u16>(inst)};
            return {static_cast<u16>(inst)};
        }
    }
}

template<typename TestEnv>
Dynarmic::A32::UserConfig GetUserConfig(TestEnv& testenv) {
    Dynarmic::A32::UserConfig user_config;
    user_config.optimizations &= ~OptimizationFlag::FastDispatch;
    user_config.callbacks = &testenv;
    user_config.always_little_endian = true;
    return user_config;
}

// Everything one fuzz instance depends on; printed as an "a32-replay" line on a mismatch.
template<typename TestEnv>
struct A32FuzzCase {
    typename A32Unicorn<TestEnv>::RegisterArray regs{};
    typename A32Unicorn<TestEnv>::ExtRegArray vecs{};
    std::vector<typename TestEnv::InstructionType> instructions;
    u32 cpsr = 0;
    u32 fpscr = 0;
    size_t ticks_left = 0;
};

template<typename TestEnv>
constexpr bool is_thumb_env = std::is_same_v<TestEnv, ThumbTestEnv>;

template<typename TestEnv>
std::string ToReplayLine(const A32FuzzCase<TestEnv>& c) {
    std::string line = fmt::format("a32-replay mode={} cpsr={:x} fpscr={:x} ticks={} insts=", is_thumb_env<TestEnv> ? "thumb" : "arm", c.cpsr, c.fpscr, c.ticks_left);
    for (size_t i = 0; i < c.instructions.size(); ++i)
        line += fmt::format("{}{:0{}x}", i ? "," : "", c.instructions[i], sizeof(c.instructions[i]) * 2);
    line += " r=";
    for (size_t i = 0; i < c.regs.size(); ++i)
        line += fmt::format("{}{:x}", i ? "," : "", c.regs[i]);
    line += " s=";
    for (size_t i = 0; i < c.vecs.size(); ++i)
        line += fmt::format("{}{:x}", i ? "," : "", c.vecs[i]);
    return line;
}

template<typename TestEnv>
std::optional<A32FuzzCase<TestEnv>> FromReplayLine(const std::string& line) {
    std::istringstream in{line};
    std::string token;
    if (!(in >> token) || token != "a32-replay")
        return std::nullopt;
    const auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream parts{s};
        for (std::string part; std::getline(parts, part, ',');)
            out.push_back(part);
        return out;
    };
    const auto hex = [](const std::string& s) { return std::strtoull(s.c_str(), nullptr, 16); };
    A32FuzzCase<TestEnv> c;
    while (in >> token) {
        const size_t eq = token.find('=');
        if (eq == std::string::npos)
            return std::nullopt;
        const std::string key = token.substr(0, eq);
        const std::string value = token.substr(eq + 1);
        if (key == "mode") {
            if (value != (is_thumb_env<TestEnv> ? "thumb" : "arm"))
                return std::nullopt;
        } else if (key == "cpsr") {
            c.cpsr = u32(hex(value));
        } else if (key == "fpscr") {
            c.fpscr = u32(hex(value));
        } else if (key == "ticks") {
            c.ticks_left = std::strtoull(value.c_str(), nullptr, 10);
        } else if (key == "insts") {
            for (const auto& s : split(value))
                c.instructions.push_back(static_cast<typename TestEnv::InstructionType>(hex(s)));
        } else if (key == "r") {
            const auto rs = split(value);
            for (size_t i = 0; i < rs.size() && i < c.regs.size(); ++i)
                c.regs[i] = u32(hex(rs[i]));
        } else if (key == "s") {
            const auto ss = split(value);
            for (size_t i = 0; i < ss.size() && i < c.vecs.size(); ++i)
                c.vecs[i] = u32(hex(ss[i]));
        }
    }
    if (c.instructions.empty())
        return std::nullopt;
    return c;
}

std::string_view ArmName(u32 instruction) {
    if (auto const name = A32::GetNameVFP<A32::TranslatorVisitor>(instruction))
        return *name;
    if (auto const name = A32::GetNameASIMD<A32::TranslatorVisitor>(instruction))
        return *name;
    if (auto const name = A32::GetNameArm<A32::TranslatorVisitor>(instruction))
        return *name;
    return "<unallocated>";
}

// Thumb names, looked up in the order translate_thumb.cpp decodes them.
std::string_view ThumbName(u32 instruction, bool is_four_bytes) {
    if (!is_four_bytes) {
        auto const name = A32::GetNameThumb16<A32::TranslatorVisitor>(u16(instruction));
        return name ? *name : std::string_view{"<unallocated>"};
    }
    if ((instruction & 0xEC000000) == 0xEC000000 || (instruction & 0xFF100000) == 0xF9000000) {
        if (auto const name = A32::GetNameVFP<A32::TranslatorVisitor>(instruction))
            return *name;
        u32 arm = 0xF7F0A000;
        if ((instruction & 0xEF000000) == 0xEF000000)
            arm = 0xF2000000 | (mcl::bit::get_bit<28>(instruction) << 24) | (instruction & 0x00FFFFFF);
        else if ((instruction & 0xFF000000) == 0xF9000000)
            arm = 0xF4000000 | (instruction & 0x00FFFFFF);
        if (auto const name = A32::GetNameASIMD<A32::TranslatorVisitor>(arm))
            return *name;
    }
    auto const name = A32::GetNameThumb32<A32::TranslatorVisitor>(instruction);
    return name ? *name : std::string_view{"<unallocated>"};
}

template<typename TestEnv>
std::vector<std::pair<u32, std::string>> ListInstructions(const A32FuzzCase<TestEnv>& c) {
    std::vector<std::pair<u32, std::string>> out;
    if constexpr (is_thumb_env<TestEnv>) {
        for (size_t i = 0; i < c.instructions.size(); ++i) {
            const u16 first = c.instructions[i];
            const bool is_four_bytes = (first & 0xF800) == 0xE800 || (first & 0xF800) == 0xF000 || (first & 0xF800) == 0xF800;
            if (is_four_bytes && i + 1 < c.instructions.size()) {
                const u32 inst = (u32(first) << 16) | c.instructions[i + 1];
                out.emplace_back(inst, std::string{ThumbName(inst, true)});
                ++i;
            } else {
                out.emplace_back(first, std::string{ThumbName(first, false)});
            }
        }
    } else {
        for (const u32 inst : c.instructions)
            out.emplace_back(inst, std::string{ArmName(inst)});
    }
    return out;
}

template<typename TestEnv>
void DumpMismatch(Dynarmic::A32::Jit& jit, A32Unicorn<TestEnv>& uni, TestEnv& jit_env, TestEnv& uni_env, const A32FuzzCase<TestEnv>& c, size_t iteration, const std::string& diff, bool full) {
    fmt::print("=== fuzz mismatch: test=\"{}\" seed={} iteration={} diff={}\n", Catch::getResultCapture().getCurrentTestName(), Catch::getSeed(), iteration, diff);
    const auto listing = ListInstructions(c);
    for (const auto& [inst, name] : listing)
        fmt::print("  inst {:08x} {}\n", inst, name);
    if (listing.size() == 1 || (is_thumb_env<TestEnv> && listing.size() == 2 && listing[0].first == 0xbf00)) {
        // Initial values of the registers the usual field positions can name.
        const u32 inst = listing.back().first;
        const std::array<std::pair<const char*, u32>, 4> fields{{{"12", (inst >> 12) & 15}, {"16", (inst >> 16) & 15}, {"0", inst & 15}, {"8", (inst >> 8) & 15}}};
        for (const auto& [bit, index] : fields)
            fmt::print("  field@{}: r{} {:08x}  d{} {:08x}{:08x}\n", bit, index, c.regs[index], index, c.vecs[index * 2 + 1], c.vecs[index * 2]);
    }
    const FP::FPCR fpcr{c.fpscr};
    fmt::print("  cpsr {:08x} fpscr {:08x} (AHP {} DN {} FZ {} RMode {} FZ16 {})\n", c.cpsr, c.fpscr, fpcr.AHP(), fpcr.DN(), fpcr.FZ(), static_cast<size_t>(fpcr.RMode()), fpcr.FZ16());
    fmt::print("  {:>5} {:>8} {:>8} {:>8}\n", "", "initial", "unicorn", "dynarmic");
    const auto uni_regs = uni.GetRegisters();
    for (size_t i = 0; i < c.regs.size(); ++i) {
        if (uni_regs[i] != jit.Regs()[i])
            fmt::print("  {:>5} {:08x} {:08x} {:08x} *\n", fmt::format("r{}", i), c.regs[i], uni_regs[i], jit.Regs()[i]);
    }
    const auto uni_ext_regs = uni.GetExtRegs();
    for (size_t i = 0; i < c.vecs.size(); ++i) {
        if (uni_ext_regs[i] != jit.ExtRegs()[i])
            fmt::print("  {:>5} {:08x} {:08x} {:08x} *\n", fmt::format("s{}", i), c.vecs[i], uni_ext_regs[i], jit.ExtRegs()[i]);
    }
    fmt::print("  cpsr -> {:08x} {:08x}, fpscr -> {:08x} {:08x}\n", uni.GetCpsr(), jit.Cpsr(), uni.GetFpscr(), jit.Fpscr());
    std::map<u32, std::pair<int, int>> memory;
    for (const auto& [addr, value] : uni_env.modified_memory)
        memory[addr].first = value + 1;
    for (const auto& [addr, value] : jit_env.modified_memory)
        memory[addr].second = value + 1;
    for (const auto& [addr, values] : memory) {
        if (values.first != values.second)
            fmt::print("  mem {:08x}: {:>2} {:>2} *\n", addr, values.first ? fmt::format("{:02x}", values.first - 1) : "--", values.second ? fmt::format("{:02x}", values.second - 1) : "--");
    }
    for (const auto& interrupt : uni_env.interrupts)
        fmt::print("  unicorn: {}\n", interrupt);
    for (const auto& exception : jit_env.exceptions)
        fmt::print("  dynarmic: {}\n", exception);
    if (full) {
        fmt::print("{}", Dynarmic::Common::DisassembleAArch32(is_thumb_env<TestEnv>, c.regs[15], reinterpret_cast<const u8*>(c.instructions.data()), c.instructions.size() * sizeof(c.instructions[0])));
        if (std::getenv("DYNARMIC_FUZZ_DUMP_X64"))
            fmt::print("x86_64:\n{}\n", jit.Disassemble());
    }
    fmt::print("{}\n", ToReplayLine(c));
    std::fflush(stdout);
}

// Runs one case on both and returns whether they agree; a disagreement is reported as asked.
template<typename TestEnv>
bool RunFuzzCase(Dynarmic::A32::Jit& jit, A32Unicorn<TestEnv>& uni, TestEnv& jit_env, TestEnv& uni_env, const A32FuzzCase<TestEnv>& c, size_t iteration, FuzzReport report) {
    const u32 initial_pc = c.regs[15];
    const u32 num_words = initial_pc / sizeof(typename TestEnv::InstructionType);
    const u32 code_mem_size = num_words + static_cast<u32>(c.instructions.size());
    const u32 expected_end_pc = code_mem_size * sizeof(typename TestEnv::InstructionType);

    jit_env.code_mem.resize(code_mem_size);
    uni_env.code_mem.resize(code_mem_size);
    std::fill(jit_env.code_mem.begin(), jit_env.code_mem.end(), TestEnv::infinite_loop);
    std::fill(uni_env.code_mem.begin(), uni_env.code_mem.end(), TestEnv::infinite_loop);

    std::copy(c.instructions.begin(), c.instructions.end(), jit_env.code_mem.begin() + num_words);
    std::copy(c.instructions.begin(), c.instructions.end(), uni_env.code_mem.begin() + num_words);
    jit_env.PadCodeMem();
    uni_env.PadCodeMem();
    jit_env.modified_memory.clear();
    uni_env.modified_memory.clear();
    jit_env.interrupts.clear();
    uni_env.interrupts.clear();
    jit_env.record_exceptions = true;
    jit_env.exceptions.clear();
    jit_env.code_mem_modified_by_guest = false;
    uni_env.code_mem_modified_by_guest = false;

    jit.Regs() = c.regs;
    jit.ExtRegs() = c.vecs;
    jit.SetFpscr(c.fpscr);
    jit.SetCpsr(c.cpsr);
    jit.ClearCache();
    // CPSR first: Unicorn 2 starts in SVC mode, so r13/r14 written before the switch to USR
    // would land in the SVC bank.
    uni.SetCpsr(c.cpsr);
    uni.SetRegisters(c.regs);
    uni.SetExtRegs(c.vecs);
    uni.SetFpscr(c.fpscr);
    uni.EnableFloatingPointAccess();
    uni.ClearPageCache();

    jit_env.ticks_left = c.ticks_left;
    CheckedRun([&]() { jit.Run(); });

    // Unicorn 1 counted Thumb halfwords; Unicorn 2 steps one instruction per call, as the JIT counts.
    uni_env.ticks_left = c.ticks_left;
    uni.Run();

    if (uni_env.code_mem_modified_by_guest != jit_env.code_mem_modified_by_guest) {
        if (report != FuzzReport::Quiet)
            DumpMismatch(jit, uni, jit_env, uni_env, c, iteration, "code_mem_modified", report == FuzzReport::Full);
        return false;
    }
    if (uni_env.code_mem_modified_by_guest)
        return true;

    // Qemu doesn't do Thumb transitions??
    {
        const u32 uni_pc = uni.GetPC();
        const bool is_thumb = (jit.Cpsr() & (1 << 5)) != 0;
        const u32 new_uni_pc = uni_pc & (is_thumb ? 0xFFFFFFFE : 0xFFFFFFFC);
        uni.SetPC(new_uni_pc);
    }

    if (uni.GetRegisters()[15] > jit.Regs()[15]) {
        int trials = 0;
        while (jit.Regs()[15] >= initial_pc && jit.Regs()[15] < expected_end_pc && trials++ < 100 && uni.GetRegisters()[15] != jit.Regs()[15]) {
            fmt::print("Warning: Possible unicorn overrrun, attempt recovery\n");
            jit.Step();
        }
    }

    // TODO: Why the difference? QEMU what are you doing???
    jit.Regs()[15] = uni.GetRegisters()[15];

    std::string diff;
    const auto note = [&diff](bool equal, const char* what) {
        if (!equal)
            diff += diff.empty() ? what : fmt::format(",{}", what);
    };
    note(uni.GetRegisters() == jit.Regs(), "regs");
    note(uni.GetExtRegs() == jit.ExtRegs(), "vecs");
    note((uni.GetCpsr() & 0xFFFFFDDF) == (jit.Cpsr() & 0xFFFFFDDF), "cpsr");
    note((uni.GetFpscr() & 0xF8000000) == (jit.Fpscr() & 0xF8000000), "fpscr");
    note(uni_env.modified_memory == jit_env.modified_memory, "mem");
    // Both may stop on the same undefined encoding; only one of them stopping is a mismatch.
    note(uni_env.interrupts.empty() == jit_env.exceptions.empty(), "interrupt");
    if (diff.empty())
        return true;

    if (report != FuzzReport::Quiet)
        DumpMismatch(jit, uni, jit_env, uni_env, c, iteration, diff, report == FuzzReport::Full);
    return false;
}

template<typename TestEnv>
static bool RunTestInstance(Dynarmic::A32::Jit& jit,
                            A32Unicorn<TestEnv>& uni,
                            TestEnv& jit_env,
                            TestEnv& uni_env,
                            const typename A32Unicorn<TestEnv>::RegisterArray& regs,
                            const typename A32Unicorn<TestEnv>::ExtRegArray& vecs,
                            const std::vector<typename TestEnv::InstructionType>& instructions,
                            const u32 cpsr,
                            const u32 fpscr,
                            const size_t ticks_left,
                            size_t iteration) {
    A32FuzzCase<TestEnv> c;
    c.regs = regs;
    c.vecs = vecs;
    c.instructions = instructions;
    c.cpsr = cpsr;
    c.fpscr = fpscr;
    c.ticks_left = ticks_left;
    const bool ok = RunFuzzCase(jit, uni, jit_env, uni_env, c, iteration, FuzzKeepGoing() ? FuzzReport::Brief : FuzzReport::Full);
    if (!ok && !FuzzKeepGoing())
        FAIL("dynarmic and Unicorn disagree, see the fuzz mismatch above");
    return ok;
}

template<typename TestEnv>
void ReplayFuzzCases(const char* prefix_mode) {
    const char* replay = std::getenv("DYNARMIC_FUZZ_REPLAY");
    if (!replay)
        SKIP("set DYNARMIC_FUZZ_REPLAY to a replay line or a file of them");
    std::vector<std::string> lines;
    const auto wanted = [prefix_mode](const std::string& line) {
        return line.starts_with("a32-replay") && line.find(prefix_mode) != std::string::npos;
    };
    if (std::string{replay}.starts_with("a32-replay")) {
        if (wanted(replay))
            lines.emplace_back(replay);
    } else {
        std::ifstream file{replay};
        for (std::string line; std::getline(file, line);) {
            if (wanted(line))
                lines.push_back(line);
        }
    }
    if (lines.empty())
        SKIP("no " << prefix_mode << " replay lines");

    TestEnv jit_env{};
    TestEnv uni_env{};
    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<TestEnv> uni{uni_env};

    size_t mismatches = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto c = FromReplayLine<TestEnv>(lines[i]);
        REQUIRE(c.has_value());
        if (RunFuzzCase(jit, uni, jit_env, uni_env, *c, i, FuzzReport::Full))
            continue;
        ++mismatches;
        if (!std::getenv("DYNARMIC_FUZZ_MINIMIZE") || c->instructions.size() < 2)
            continue;
        // NOP halfword by halfword in Thumb, so a 32-bit instruction becomes two NOPs.
        constexpr auto nop = is_thumb_env<TestEnv> ? 0xbf00 : 0xe1a00000;  // NOP, MOV r0, r0
        A32FuzzCase<TestEnv> minimized = *c;
        for (size_t j = 0; j < minimized.instructions.size(); ++j) {
            auto candidate = minimized;
            const bool is_four_bytes = is_thumb_env<TestEnv> && (candidate.instructions[j] & 0xF800) >= 0xE800;
            candidate.instructions[j] = static_cast<typename TestEnv::InstructionType>(nop);
            if (is_four_bytes && j + 1 < candidate.instructions.size())
                candidate.instructions[j + 1] = static_cast<typename TestEnv::InstructionType>(nop);
            if (!RunFuzzCase(jit, uni, jit_env, uni_env, candidate, i, FuzzReport::Quiet))
                minimized = candidate;
            if (is_four_bytes)
                ++j;
        }
        fmt::print("minimized:\n");
        RunFuzzCase(jit, uni, jit_env, uni_env, minimized, i, FuzzReport::Full);
    }
    REQUIRE(mismatches == 0);
}
}  // Anonymous namespace

TEST_CASE("A32: Single random arm instruction", "[arm][unicorn]") {
    ArmTestEnv jit_env{};
    ArmTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ArmTestEnv> uni{uni_env};

    A32Unicorn<ArmTestEnv>::RegisterArray regs;
    A32Unicorn<ArmTestEnv>::ExtRegArray ext_reg;
    std::vector<u32> instructions(1);

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x10;
        const u32 fpcr = RandomFpcr();

        instructions[0] = GenRandomArmInst(start_address, true);

        INFO("Instruction: 0x" << std::hex << instructions[0]);

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 1, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Small random arm block", "[arm][unicorn]") {
    ArmTestEnv jit_env{};
    ArmTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ArmTestEnv> uni{uni_env};

    A32Unicorn<ArmTestEnv>::RegisterArray regs;
    A32Unicorn<ArmTestEnv>::ExtRegArray ext_reg;
    std::vector<u32> instructions(5);

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x10;
        const u32 fpcr = RandomFpcr();

        instructions[0] = GenRandomArmInst(start_address + 0, false);
        instructions[1] = GenRandomArmInst(start_address + 4, false);
        instructions[2] = GenRandomArmInst(start_address + 8, false);
        instructions[3] = GenRandomArmInst(start_address + 12, false);
        instructions[4] = GenRandomArmInst(start_address + 16, true);

        INFO("Instruction 1: 0x" << std::hex << instructions[0]);
        INFO("Instruction 2: 0x" << std::hex << instructions[1]);
        INFO("Instruction 3: 0x" << std::hex << instructions[2]);
        INFO("Instruction 4: 0x" << std::hex << instructions[3]);
        INFO("Instruction 5: 0x" << std::hex << instructions[4]);

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 5, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Large random arm block", "[arm][unicorn]") {
    ArmTestEnv jit_env{};
    ArmTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ArmTestEnv> uni{uni_env};

    A32Unicorn<ArmTestEnv>::RegisterArray regs;
    A32Unicorn<ArmTestEnv>::ExtRegArray ext_reg;

    constexpr size_t instruction_count = 100;
    std::vector<u32> instructions(instruction_count);

    const size_t iterations = FuzzIterations(10000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u64 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x10;
        const u32 fpcr = RandomFpcr();

        for (size_t j = 0; j < instruction_count; ++j) {
            instructions[j] = GenRandomArmInst(start_address + j * 4, j == instruction_count - 1);
        }

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 100, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Single random thumb instruction", "[thumb][unicorn]") {
    ThumbTestEnv jit_env{};
    ThumbTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ThumbTestEnv> uni{uni_env};

    A32Unicorn<ThumbTestEnv>::RegisterArray regs;
    A32Unicorn<ThumbTestEnv>::ExtRegArray ext_reg;
    std::vector<u16> instructions;

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x1F0;
        const u32 fpcr = RandomFpcr();

        instructions = GenRandomThumbInst(start_address, true);

        INFO("Instruction: 0x" << std::hex << instructions[0]);

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 1, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Single random thumb instruction (offset)", "[thumb][unicorn]") {
    ThumbTestEnv jit_env{};
    ThumbTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ThumbTestEnv> uni{uni_env};

    A32Unicorn<ThumbTestEnv>::RegisterArray regs;
    A32Unicorn<ThumbTestEnv>::ExtRegArray ext_reg;
    std::vector<u16> instructions;

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x1F0;
        const u32 fpcr = RandomFpcr();

        instructions.clear();
        instructions.push_back(0xbf00);  // NOP
        const std::vector<u16> inst = GenRandomThumbInst(start_address + 2, true);
        instructions.insert(instructions.end(), inst.begin(), inst.end());

        INFO("Instruction: 0x" << std::hex << inst[0]);

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 2, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Small random thumb block", "[thumb][unicorn]") {
    ThumbTestEnv jit_env{};
    ThumbTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ThumbTestEnv> uni{uni_env};

    A32Unicorn<ThumbTestEnv>::RegisterArray regs;
    A32Unicorn<ThumbTestEnv>::ExtRegArray ext_reg;
    std::vector<u16> instructions;

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x1F0;
        const u32 fpcr = RandomFpcr();

        instructions.clear();
        for (size_t i = 0; i < 5; i++) {
            const std::vector<u16> inst = GenRandomThumbInst(start_address + instructions.size() * 2, i == 4);
            instructions.insert(instructions.end(), inst.begin(), inst.end());
        }

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, 5, iteration);
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("A32: Test thumb IT instruction", "[thumb][unicorn]") {
    ThumbTestEnv jit_env{};
    ThumbTestEnv uni_env{};

    Dynarmic::A32::Jit jit{GetUserConfig(jit_env)};
    A32Unicorn<ThumbTestEnv> uni{uni_env};

    A32Unicorn<ThumbTestEnv>::RegisterArray regs;
    A32Unicorn<ThumbTestEnv>::ExtRegArray ext_reg;
    std::vector<u16> instructions;

    const size_t iterations = FuzzIterations(100000);
    size_t mismatches = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::generate(regs.begin(), regs.end(), [] { return RandInt<u32>(0, ~u32(0)); });
        std::generate(ext_reg.begin(), ext_reg.end(), [] { return RandInt<u32>(0, ~u32(0)); });

        const size_t pre_instructions = RandInt<size_t>(0, 3);
        const size_t post_instructions = RandInt<size_t>(5, 8);

        const u32 start_address = 100;
        const u32 cpsr = (RandInt<u32>(0, 0xF) << 28) | 0x1F0;
        const u32 fpcr = RandomFpcr();

        instructions.clear();

        for (size_t i = 0; i < pre_instructions; i++) {
            const std::vector<u16> inst = GenRandomThumbInst(start_address + instructions.size() * 2, false);
            instructions.insert(instructions.end(), inst.begin(), inst.end());
        }

        // Emit IT instruction
        A32::ITState it_state = [&] {
            while (true) {
                const u16 imm8 = RandInt<u16>(0, 0xFF);
                if (mcl::bit::get_bits<0, 3>(imm8) == 0b0000 || mcl::bit::get_bits<4, 7>(imm8) == 0b1111 || (mcl::bit::get_bits<4, 7>(imm8) == 0b1110 && std::popcount(mcl::bit::get_bits<0, 3>(imm8)) != 1)) {
                    continue;
                }
                instructions.push_back(0b1011111100000000 | imm8);
                return A32::ITState{static_cast<u8>(imm8)};
            }
        }();

        for (size_t i = 0; i < post_instructions; i++) {
            const std::vector<u16> inst = GenRandomThumbInst(start_address + instructions.size() * 2, i == post_instructions - 1, it_state);
            instructions.insert(instructions.end(), inst.begin(), inst.end());
            it_state = it_state.Advance();
        }

        regs[15] = start_address;
        mismatches += !RunTestInstance(jit, uni, jit_env, uni_env, regs, ext_reg, instructions, cpsr, fpcr, pre_instructions + 1 + post_instructions, iteration);
    }
    REQUIRE(mismatches == 0);
}

// DYNARMIC_FUZZ_REPLAY holds one "a32-replay ..." line, or names a file of them.
TEST_CASE("A32: Replay fuzz cases", "[.][arm][thumb][unicorn]") {
    SECTION("arm") {
        ReplayFuzzCases<ArmTestEnv>("mode=arm");
    }
    SECTION("thumb") {
        ReplayFuzzCases<ThumbTestEnv>("mode=thumb");
    }
}
