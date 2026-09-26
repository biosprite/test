// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <stdexcept>
#include <type_traits>
#include <fmt/format.h>
#include "dynarmic/mcl/bit.hpp"
#include "dynarmic/tests/unicorn_emu/a32_unicorn.h"
#include "dynarmic/tests/A32/testenv.h"

// Throws, so a failing Unicorn call fails the test instead of a soft assert nobody sees.
#define CHECKED(expr)                                                                         \
    do {                                                                                      \
        if (auto const cerr_ = (expr))                                                        \
            throw std::runtime_error(fmt::format("Call " #expr " failed with error {} ({})", \
                                                 static_cast<int>(cerr_), uc_strerror(cerr_))); \
    } while (0)

constexpr u32 BEGIN_ADDRESS = 0;
constexpr u32 END_ADDRESS = ~u32(0);

template<class TestEnvironment>
A32Unicorn<TestEnvironment>::A32Unicorn(TestEnvironment& testenv)
        : testenv{testenv} {
    constexpr uc_mode open_mode = std::is_same_v<TestEnvironment, ArmTestEnv> ? UC_MODE_ARM : UC_MODE_THUMB;

    CHECKED(uc_open(UC_ARCH_ARM, open_mode, &uc));
    // Unicorn 2 defaults to a Cortex-A15 (ARMv7); "max" adds the ARMv8 AArch32 instructions.
    CHECKED(uc_ctl_set_cpu_model(uc, UC_CPU_ARM_MAX));
    CHECKED(uc_hook_add(uc, &intr_hook, UC_HOOK_INTR, (void*)InterruptHook, this, BEGIN_ADDRESS, END_ADDRESS));
    CHECKED(uc_hook_add(uc, &mem_invalid_hook, UC_HOOK_MEM_INVALID, (void*)UnmappedMemoryHook, this, BEGIN_ADDRESS, END_ADDRESS));
    CHECKED(uc_hook_add(uc, &mem_write_prot_hook, UC_HOOK_MEM_WRITE, (void*)MemoryWriteHook, this, BEGIN_ADDRESS, END_ADDRESS));
}

template<class TestEnvironment>
A32Unicorn<TestEnvironment>::~A32Unicorn() {
    for (const auto& page : pages)
        uc_mem_unmap(uc, page->address, 4096);
    uc_hook_del(uc, intr_hook);
    uc_hook_del(uc, mem_invalid_hook);
    uc_hook_del(uc, mem_write_prot_hook);
    uc_close(uc);
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::RunSteps() {
    // Thumb execution mode requires the LSB to be set to 1.
    constexpr u64 pc_mask = std::is_same_v<TestEnvironment, ArmTestEnv> ? 0 : 1;
    while (testenv.ticks_left > 0) {
        const u32 pc = GetPC() | pc_mask;
        if (!testenv.IsInCodeMem(pc)) {
            return;
        }
        if (auto cerr_ = uc_emu_start(uc, pc, END_ADDRESS, 0, 1)) {
            // An UNDEFINED instruction stops Unicorn here; report it like an interrupt.
            testenv.interrupts.emplace_back(fmt::format("uc_emu_start failed @ {:08x} (code = {:08x}) with error {} ({})", pc, *testenv.MemoryReadCode(pc & ~1u), static_cast<size_t>(cerr_), uc_strerror(cerr_)));
            return;
        }
        testenv.ticks_left--;
        if (!testenv.interrupts.empty() || testenv.code_mem_modified_by_guest) {
            return;
        }
    }
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::Run() {
    // Unicorn 2.1.4 leaves ITSTATE set after an IT block in which the invalid-memory hook had
    // to map a page, so a run that mapped pages is repeated with those pages already mapped.
    const RegisterArray regs = GetRegisters();
    const ExtRegArray ext_regs = GetExtRegs();
    const u32 cpsr = GetCpsr();
    const u32 fpscr = GetFpscr();
    const u64 ticks_left = testenv.ticks_left;
    const auto modified_memory = testenv.modified_memory;
    for (size_t pass = 0;; ++pass) {
        pages_mapped_by_hook = 0;
        RunSteps();
        if (pages_mapped_by_hook == 0 || pass == 3)
            break;
        testenv.ticks_left = ticks_left;
        testenv.modified_memory = modified_memory;
        testenv.interrupts.clear();
        testenv.code_mem_modified_by_guest = false;
        std::vector<u32> addresses;
        for (const auto& page : pages)
            addresses.push_back(page->address);
        ClearPageCache();
        for (const u32 address : addresses)
            GeneratePage(address);
        SetCpsr(cpsr);
        SetRegisters(regs);
        SetExtRegs(ext_regs);
        SetFpscr(fpscr);
    }

    const bool T = mcl::bit::get_bit<5>(GetCpsr());
    const u32 new_pc = GetPC() | (T ? 1 : 0);
    SetPC(new_pc);
}

template<class TestEnvironment>
u32 A32Unicorn<TestEnvironment>::GetPC() const {
    u32 pc;
    CHECKED(uc_reg_read(uc, UC_ARM_REG_PC, &pc));
    return pc;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetPC(u32 value) {
    CHECKED(uc_reg_write(uc, UC_ARM_REG_PC, &value));
}

template<class TestEnvironment>
u32 A32Unicorn<TestEnvironment>::GetSP() const {
    u32 sp;
    CHECKED(uc_reg_read(uc, UC_ARM_REG_SP, &sp));
    return sp;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetSP(u32 value) {
    CHECKED(uc_reg_write(uc, UC_ARM_REG_SP, &value));
}

constexpr std::array<int, Unicorn::A32::num_gprs> gpr_ids{
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_R3,
    UC_ARM_REG_R4,
    UC_ARM_REG_R5,
    UC_ARM_REG_R6,
    UC_ARM_REG_R7,
    UC_ARM_REG_R8,
    UC_ARM_REG_R9,
    UC_ARM_REG_R10,
    UC_ARM_REG_R11,
    UC_ARM_REG_R12,
    UC_ARM_REG_R13,
    UC_ARM_REG_R14,
    UC_ARM_REG_R15,
};

template<class TestEnvironment>
Unicorn::A32::RegisterArray A32Unicorn<TestEnvironment>::GetRegisters() const {
    Unicorn::A32::RegisterArray regs{};
    Unicorn::A32::RegisterPtrArray ptrs;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i] = &regs[i];
    }

    CHECKED(uc_reg_read_batch(uc, const_cast<int*>(gpr_ids.data()),
                              reinterpret_cast<void**>(ptrs.data()), static_cast<int>(Unicorn::A32::num_gprs)));
    return regs;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetRegisters(const RegisterArray& value) {
    Unicorn::A32::RegisterConstPtrArray ptrs;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i] = &value[i];
    }

    CHECKED(uc_reg_write_batch(uc, const_cast<int*>(gpr_ids.data()),
                               reinterpret_cast<void**>(const_cast<u32**>(ptrs.data())), static_cast<int>(ptrs.size())));
}

using DoubleExtRegPtrArray = std::array<Unicorn::A32::ExtRegArray::pointer, Unicorn::A32::num_ext_regs / 2>;
using DoubleExtRegConstPtrArray = std::array<Unicorn::A32::ExtRegArray::const_pointer, Unicorn::A32::num_ext_regs / 2>;

constexpr std::array<int, Unicorn::A32::num_ext_regs / 2> double_ext_reg_ids{
    UC_ARM_REG_D0,
    UC_ARM_REG_D1,
    UC_ARM_REG_D2,
    UC_ARM_REG_D3,
    UC_ARM_REG_D4,
    UC_ARM_REG_D5,
    UC_ARM_REG_D6,
    UC_ARM_REG_D7,
    UC_ARM_REG_D8,
    UC_ARM_REG_D9,
    UC_ARM_REG_D10,
    UC_ARM_REG_D11,
    UC_ARM_REG_D12,
    UC_ARM_REG_D13,
    UC_ARM_REG_D14,
    UC_ARM_REG_D15,
    UC_ARM_REG_D16,
    UC_ARM_REG_D17,
    UC_ARM_REG_D18,
    UC_ARM_REG_D19,
    UC_ARM_REG_D20,
    UC_ARM_REG_D21,
    UC_ARM_REG_D22,
    UC_ARM_REG_D23,
    UC_ARM_REG_D24,
    UC_ARM_REG_D25,
    UC_ARM_REG_D26,
    UC_ARM_REG_D27,
    UC_ARM_REG_D28,
    UC_ARM_REG_D29,
    UC_ARM_REG_D30,
    UC_ARM_REG_D31,
};

template<class TestEnvironment>
Unicorn::A32::ExtRegArray A32Unicorn<TestEnvironment>::GetExtRegs() const {
    Unicorn::A32::ExtRegArray ext_regs{};
    DoubleExtRegPtrArray ptrs;
    for (size_t i = 0; i < ptrs.size(); ++i)
        ptrs[i] = &ext_regs[i * 2];

    CHECKED(uc_reg_read_batch(uc, const_cast<int*>(double_ext_reg_ids.data()),
                              reinterpret_cast<void**>(ptrs.data()), static_cast<int>(ptrs.size())));

    return ext_regs;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetExtRegs(const ExtRegArray& value) {
    DoubleExtRegConstPtrArray ptrs;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i] = &value[i * 2];
    }

    CHECKED(uc_reg_write_batch(uc, const_cast<int*>(double_ext_reg_ids.data()),
                               reinterpret_cast<void* const*>(const_cast<u32**>(ptrs.data())), static_cast<int>(ptrs.size())));
}

template<class TestEnvironment>
u32 A32Unicorn<TestEnvironment>::GetFpscr() const {
    u32 fpsr;
    CHECKED(uc_reg_read(uc, UC_ARM_REG_FPSCR, &fpsr));
    return fpsr;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetFpscr(u32 value) {
    CHECKED(uc_reg_write(uc, UC_ARM_REG_FPSCR, &value));
}

template<class TestEnvironment>
u32 A32Unicorn<TestEnvironment>::GetFpexc() const {
    u32 value = 0;
    CHECKED(uc_reg_read(uc, UC_ARM_REG_FPEXC, &value));
    return value;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetFpexc(u32 value) {
    CHECKED(uc_reg_write(uc, UC_ARM_REG_FPEXC, &value));
}

template<class TestEnvironment>
u32 A32Unicorn<TestEnvironment>::GetCpsr() const {
    u32 pstate;
    CHECKED(uc_reg_read(uc, UC_ARM_REG_CPSR, &pstate));
    return pstate;
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::SetCpsr(u32 value) {
    CHECKED(uc_reg_write(uc, UC_ARM_REG_CPSR, &value));
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::EnableFloatingPointAccess() {
    const u32 new_fpexc = GetFpexc() | (1U << 30);
    SetFpexc(new_fpexc);
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::ClearPageCache() {
    for (const auto& page : pages) {
        CHECKED(uc_mem_unmap(uc, page->address, 4096));
    }
    pages.clear();
    // Unicorn 2.1.4 fails with UC_ERR_MAP when the invalid-memory hook maps the last page of
    // the 32-bit space, so that one page is mapped up front.
    GeneratePage(0xFFFFF000);
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::DumpMemoryInformation() {
    uc_mem_region* regions;
    u32 count;
    CHECKED(uc_mem_regions(uc, &regions, &count));

    for (u32 i = 0; i < count; ++i) {
        printf("region: start 0x%08x end 0x%08x perms 0x%08x\n", static_cast<u32>(regions[i].begin), static_cast<u32>(regions[i].end), regions[i].perms);
    }

    CHECKED(uc_free(regions));
}

template<class TestEnvironment>
void A32Unicorn<TestEnvironment>::InterruptHook(uc_engine* /*uc*/, u32 int_number, void* user_data) {
    auto* this_ = static_cast<A32Unicorn*>(user_data);

    u32 esr = 0;
    // CHECKED(uc_reg_read(uc, UC_ARM_REG_ESR, &esr));

    auto ec = esr >> 26;
    auto iss = esr & 0xFFFFFF;

    switch (ec) {
    case 0x15:  // SVC
        this_->testenv.CallSVC(iss);
        break;
    default:
        this_->testenv.interrupts.emplace_back(fmt::format("Unhandled interrupt: int_number: {:#x}, esr: {:#x} (ec: {:#x}, iss: {:#x})", int_number, esr, ec, iss));
        break;
    }
}

template<class TestEnvironment>
bool A32Unicorn<TestEnvironment>::GeneratePage(u32 base_address) {
    const u32 permissions = [&]() -> u32 {
        if (base_address < testenv.code_mem.size() * sizeof(typename TestEnvironment::InstructionType)) {
            return UC_PROT_READ | UC_PROT_EXEC;
        }
        return UC_PROT_READ;
    }();

    auto page = std::make_unique<Page>();
    page->address = base_address;
    for (size_t i = 0; i < page->data.size(); ++i)
        page->data[i] = u8(testenv.MemoryRead(u32(base_address + i), sizeof(u8)));

    uc_err err = uc_mem_map_ptr(uc, base_address, page->data.size(), permissions, page->data.data());
    if (err == UC_ERR_MAP)
        return false;  // page already exists
    if (err != UC_ERR_OK) {
        // Never throw through Unicorn's C frames; the test sees this as an interrupt.
        testenv.interrupts.emplace_back(fmt::format("uc_mem_map_ptr({:#x}) failed: {}", base_address, uc_strerror(err)));
        return false;
    }

    pages.emplace_back(std::move(page));
    return true;
}

template<class TestEnvironment>
bool A32Unicorn<TestEnvironment>::UnmappedMemoryHook(uc_engine* /*uc*/, uc_mem_type /*type*/, u32 start_address, int size, u64 /*value*/, void* user_data) {
    auto* this_ = static_cast<A32Unicorn*>(user_data);

    const auto is_in_range = [](u32 addr, u32 start, u32 end) {
        if (start <= end)
            return addr >= start && addr <= end;  // fffff[tttttt]fffff
        return addr >= start || addr <= end;      // ttttt]ffffff[ttttt
    };

    const u32 start_address_page = start_address & ~u32(0xFFF);
    const u32 end_address = start_address + size - 1;

    u32 current_address = start_address_page;
    do {
        this_->pages_mapped_by_hook += this_->GeneratePage(current_address);
        current_address += 0x1000;
    } while (is_in_range(current_address, start_address_page, end_address) && current_address != start_address_page);

    return true;
}

template<class TestEnvironment>
bool A32Unicorn<TestEnvironment>::MemoryWriteHook(uc_engine* /*uc*/, uc_mem_type /*type*/, u32 start_address, int size, u64 value, void* user_data) {
    auto* this_ = static_cast<A32Unicorn*>(user_data);
    this_->testenv.MemoryWrite(start_address, value, size);
    return true;
}

template class A32Unicorn<ArmTestEnv>;
template class A32Unicorn<ThumbTestEnv>;
