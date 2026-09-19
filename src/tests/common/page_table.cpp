// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cstdint>
#include <ios>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/page_table.h"

namespace {

using Common::PageTable;
using Data = PageTable::PageEntryData::Data;

// The x64 JIT's math (emit_x64_memory.h).
u64 X64JitPointer(u64 raw) {
    const u64 masked = raw & PageTable::ATTRIBUTE_MASK;
    const int shift = 63 - static_cast<int>(PageTable::SIGN_BIT);
    return static_cast<u64>(static_cast<s64>(masked << shift) >> shift);
}

// The arm64 JIT's math (emit_arm64_memory.cpp).
u64 Arm64JitPointer(u64 raw) {
    const u64 masked = raw & PageTable::ATTRIBUTE_MASK;
    const u64 width = PageTable::SIGN_BIT + 1;
    const u64 sign = 1ULL << PageTable::SIGN_BIT;
    const u64 low = masked & ((1ULL << width) - 1);
    return (low ^ sign) - sign;
}

u64 RawEntry(bool marked, Common::PageType type, u16 block, u64 offset) {
    return std::bit_cast<u64>(Data{marked, type, block, offset});
}

// Host offsets (host_pointer - guest_address), positive and negative.
constexpr std::array<u64, 7> kOffsets{
    0x0000'7F12'3456'7000ULL,                             // typical Windows/Linux user pointer
    0x0000'0001'0000'0000ULL,                             // low allocation (FreeBSD style)
    0x0000'0000'0000'1000ULL,                             // smallest positive page
    static_cast<u64>(-0x0000'0000'0000'1000LL),           // smallest negative page
    static_cast<u64>(-0x0000'007F'FFFF'F000LL),           // backing below the 39-bit guest space
    static_cast<u64>(-0x0000'0080'0000'0000LL),           // -(1 << 39)
    (1ULL << PageTable::SIGN_BIT) - 0x1000ULL,            // largest positive value that fits
};

} // Anonymous namespace

TEST_CASE("PageTable: ExtractPointer round-trips positive and negative offsets", "[common]") {
    for (const u64 offset : kOffsets) {
        const Data data{false, Common::PageType::Memory, 0x1FF, offset};
        CHECK(PageTable::PageEntryData::ExtractPointer(data) == static_cast<uintptr_t>(offset));
        CHECK(PageTable::PageEntryData::ExtractBlock(data) == 0x1FF);
    }
}

TEST_CASE("PageTable: JIT sign extension matches ExtractPointer", "[common]") {
    for (const u64 offset : kOffsets) {
        const u64 raw = RawEntry(false, Common::PageType::Memory, 0x3FF, offset);
        const u64 expected = PageTable::PageEntryData::ExtractPointer(std::bit_cast<Data>(raw));
        INFO("offset " << std::hex << offset);
        CHECK(X64JitPointer(raw) == expected);
        CHECK(Arm64JitPointer(raw) == expected);
        CHECK(expected == offset);
    }
}

TEST_CASE("PageTable: attribute bits never leak into the pointer", "[common]") {
    // All attribute bits set, pointer zero.
    const u64 raw = RawEntry(false, Common::PageType::RasterizerCachedMemory, 0xFFFF, 0);
    REQUIRE((raw & ~PageTable::ATTRIBUTE_MASK) != 0);
    CHECK(X64JitPointer(raw) == 0);
    CHECK(Arm64JitPointer(raw) == 0);
    CHECK(PageTable::PageEntryData::ExtractPointer(std::bit_cast<Data>(raw)) == 0);

    // A marked entry reads as null unless explicitly ignored.
    const Data marked{true, Common::PageType::Memory, 1, 0x0000'7000'0000'0000ULL};
    CHECK(PageTable::PageEntryData::ExtractPointer(marked) == 0);
    CHECK(PageTable::PageEntryData::ExtractPointer(marked, true) == 0x0000'7000'0000'0000ULL);
}
