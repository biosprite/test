/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <optional>

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

static bool ExclusiveSharedDecodeAndOperation(TranslatorVisitor& v, bool pair, size_t size, bool L, bool o0, std::optional<Reg> Rs, std::optional<Reg> Rt2, Reg Rn, Reg Rt) {
    // Shared Decode

    const auto acctype = o0 ? IR::AccType::ORDERED : IR::AccType::ATOMIC;
    const auto memop = L ? IR::MemOp::LOAD : IR::MemOp::STORE;
    const size_t elsize = 8 << size;
    const size_t regsize = elsize == 64 ? 64 : 32;
    const size_t datasize = pair ? elsize * 2 : elsize;

    // Operation

    const size_t dbytes = datasize / 8;

    if (memop == IR::MemOp::LOAD && pair && Rt == *Rt2) {
        return v.UnpredictableInstruction();
    } else if (memop == IR::MemOp::STORE && (*Rs == Rt || (pair && *Rs == *Rt2))) {
        if (!v.options.define_unpredictable_behaviour) {
            return v.UnpredictableInstruction();
        }
        // UNPREDICTABLE: The Constraint_NONE case is executed.
    } else if (memop == IR::MemOp::STORE && *Rs == Rn && Rn != Reg::R31) {
        return v.UnpredictableInstruction();
    }

    IR::U64 address;
    if (Rn == Reg::SP) {
        // TODO: Check SP Alignment
        address = v.SP(64);
    } else {
        address = v.X(64, Rn);
    }

    switch (memop) {
    case IR::MemOp::STORE: {
        IR::UAnyU128 data;
        if (pair && elsize == 64) {
            data = v.ir.Pack2x64To1x128(v.X(64, Rt), v.X(64, *Rt2));
        } else if (pair && elsize == 32) {
            data = v.ir.Pack2x32To1x64(v.X(32, Rt), v.X(32, *Rt2));
        } else {
            data = v.X(elsize, Rt);
        }
        const IR::U32 status = v.ExclusiveMem(address, dbytes, acctype, data);
        v.X(32, *Rs, status);
        break;
    }
    case IR::MemOp::LOAD: {
        const IR::UAnyU128 data = v.ExclusiveMem(address, dbytes, acctype);
        if (pair && elsize == 64) {
            v.X(64, Rt, v.ir.VectorGetElement(64, data, 0));
            v.X(64, *Rt2, v.ir.VectorGetElement(64, data, 1));
        } else if (pair && elsize == 32) {
            v.X(32, Rt, v.ir.LeastSignificantWord(data));
            v.X(32, *Rt2, v.ir.MostSignificantWord(data).result);
        } else {
            v.X(regsize, Rt, v.ZeroExtend(data, regsize));
        }
        break;
    }
    default:
        UNREACHABLE();
    }

    return true;
}

bool TranslatorVisitor::STXR(Imm<2> sz, Reg Rs, Reg Rn, Reg Rt) {
    const bool pair = false;
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 0;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, Rs, {}, Rn, Rt);
}

bool TranslatorVisitor::STLXR(Imm<2> sz, Reg Rs, Reg Rn, Reg Rt) {
    const bool pair = false;
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 1;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, Rs, {}, Rn, Rt);
}

bool TranslatorVisitor::STXP(Imm<1> sz, Reg Rs, Reg Rt2, Reg Rn, Reg Rt) {
    const bool pair = true;
    const size_t size = concatenate(Imm<1>{1}, sz).ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 0;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, Rs, Rt2, Rn, Rt);
}

bool TranslatorVisitor::STLXP(Imm<1> sz, Reg Rs, Reg Rt2, Reg Rn, Reg Rt) {
    const bool pair = true;
    const size_t size = concatenate(Imm<1>{1}, sz).ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 1;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, Rs, Rt2, Rn, Rt);
}

bool TranslatorVisitor::LDXR(Imm<2> sz, Reg Rn, Reg Rt) {
    const bool pair = false;
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 0;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, {}, {}, Rn, Rt);
}

bool TranslatorVisitor::LDAXR(Imm<2> sz, Reg Rn, Reg Rt) {
    const bool pair = false;
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 1;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, {}, {}, Rn, Rt);
}

bool TranslatorVisitor::LDXP(Imm<1> sz, Reg Rt2, Reg Rn, Reg Rt) {
    const bool pair = true;
    const size_t size = concatenate(Imm<1>{1}, sz).ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 0;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, {}, Rt2, Rn, Rt);
}

bool TranslatorVisitor::LDAXP(Imm<1> sz, Reg Rt2, Reg Rn, Reg Rt) {
    const bool pair = true;
    const size_t size = concatenate(Imm<1>{1}, sz).ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 1;
    return ExclusiveSharedDecodeAndOperation(*this, pair, size, L, o0, {}, Rt2, Rn, Rt);
}

static bool OrderedSharedDecodeAndOperation(TranslatorVisitor& v, size_t size, bool L, bool o0, Reg Rn, Reg Rt) {
    // Shared Decode

    const auto acctype = !o0 ? IR::AccType::LIMITEDORDERED : IR::AccType::ORDERED;
    const auto memop = L ? IR::MemOp::LOAD : IR::MemOp::STORE;
    const size_t elsize = 8 << size;
    const size_t regsize = elsize == 64 ? 64 : 32;
    const size_t datasize = elsize;

    // Operation

    const size_t dbytes = datasize / 8;

    IR::U64 address;
    if (Rn == Reg::SP) {
        // TODO: Check SP Alignment
        address = v.SP(64);
    } else {
        address = v.X(64, Rn);
    }

    switch (memop) {
    case IR::MemOp::STORE: {
        const IR::UAny data = v.X(datasize, Rt);
        v.Mem(address, dbytes, acctype, data);
        break;
    }
    case IR::MemOp::LOAD: {
        const IR::UAny data = v.Mem(address, dbytes, acctype);
        v.X(regsize, Rt, v.ZeroExtend(data, regsize));
        break;
    }
    default:
        UNREACHABLE();
    }

    return true;
}

bool TranslatorVisitor::STLLR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 0;
    return OrderedSharedDecodeAndOperation(*this, size, L, o0, Rn, Rt);
}

bool TranslatorVisitor::STLR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 0;
    const bool o0 = 1;
    return OrderedSharedDecodeAndOperation(*this, size, L, o0, Rn, Rt);
}

bool TranslatorVisitor::LDLAR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 0;
    return OrderedSharedDecodeAndOperation(*this, size, L, o0, Rn, Rt);
}

bool TranslatorVisitor::LDAR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const bool L = 1;
    const bool o0 = 1;
    return OrderedSharedDecodeAndOperation(*this, size, L, o0, Rn, Rt);
}

static Reg AtomicNextReg(Reg r) {
    return static_cast<Reg>(static_cast<size_t>(r) + 1);
}

static IR::U64 AtomicAddress(TranslatorVisitor& v, Reg Rn) {
    return Rn == Reg::SP ? IR::U64{v.SP(64)} : IR::U64{v.X(64, Rn)};
}

static IR::U64 AtomicMaskIfZero(TranslatorVisitor& v, IR::U64 value) {
    const IR::U64 nonzero = v.ir.LogicalShiftRight(v.ir.Or(value, v.ir.Sub(v.ir.Imm64(0), value)), v.ir.Imm8(63));
    return v.ir.Sub(nonzero, v.ir.Imm64(1));
}

static IR::U64 AtomicSelect(TranslatorVisitor& v, IR::U64 mask, IR::U64 if_set, IR::U64 if_clear) {
    return v.ir.Or(v.ir.And(if_set, mask), v.ir.AndNot(if_clear, mask));
}

static IR::UAny AtomicTruncate(TranslatorVisitor& v, IR::U64 value, size_t bytes) {
    switch (bytes) {
    case 1:
        return v.ir.LeastSignificantByte(value);
    case 2:
        return v.ir.LeastSignificantHalf(value);
    case 4:
        return v.ir.LeastSignificantWord(value);
    default:
        return value;
    }
}

static IR::U64 AtomicLoadZext(TranslatorVisitor& v, IR::U64 address, size_t bytes, IR::AccType acctype) {
    const IR::UAny value = v.ExclusiveMem(address, bytes, acctype);
    return bytes == 8 ? IR::U64{value} : v.ir.ZeroExtendToLong(value);
}

static IR::U64 AtomicSignExtend(TranslatorVisitor& v, IR::U64 value, size_t bytes) {
    return bytes == 8 ? value : v.ir.SignExtendToLong(AtomicTruncate(v, value, bytes));
}

static IR::U64 AtomicZeroExtendReg(TranslatorVisitor& v, Reg r, size_t bytes) {
    const IR::U64 value = v.X(64, r);
    return bytes == 8 ? value : v.ir.ZeroExtendToLong(AtomicTruncate(v, value, bytes));
}

static void AtomicCommitReg(TranslatorVisitor& v, Reg r, IR::U64 ok_mask, IR::U64 value) {
    if (r == Reg::ZR) {
        return;
    }
    v.X(64, r, AtomicSelect(v, ok_mask, value, v.X(64, r)));
}

// Continues with the next instruction on success, retries this one if the exclusive store failed.
static bool AtomicFinish(TranslatorVisitor& v, IR::U32 status) {
    v.ir.SetCheckBit(v.ir.IsZero(status));
    v.ir.SetTerm(IR::Term::CheckBit{IR::Term::LinkBlock{v.ir.current_location->AdvancePC(4)},
                                    IR::Term::LinkBlock{*v.ir.current_location}});
    return false;
}

static IR::AccType AtomicAccType(bool acquire, bool release) {
    return acquire || release ? IR::AccType::ORDERED : IR::AccType::ATOMIC;
}

bool TranslatorVisitor::AtomicMemOp(Imm<2> size, bool A, bool R, Reg Rs, Imm<4> op, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    const size_t opv = op.ZeroExtend<size_t>();
    if (opv > 0b1000) {
        return UnallocatedEncoding();
    }

    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(A, R);
    const IR::U64 old = AtomicLoadZext(*this, address, bytes, acctype);
    const IR::U64 s = AtomicZeroExtendReg(*this, Rs, bytes);

    IR::U64 result;
    switch (opv) {
    case 0b0000:
        result = ir.Add(old, s);
        break;
    case 0b0001:
        result = ir.AndNot(old, s);
        break;
    case 0b0010:
        result = ir.Eor(old, s);
        break;
    case 0b0011:
        result = ir.Or(old, s);
        break;
    case 0b0100:
        result = ir.MaxSigned(AtomicSignExtend(*this, old, bytes), AtomicSignExtend(*this, s, bytes));
        break;
    case 0b0101:
        result = ir.MinSigned(AtomicSignExtend(*this, old, bytes), AtomicSignExtend(*this, s, bytes));
        break;
    case 0b0110:
        result = ir.MaxUnsigned(old, s);
        break;
    case 0b0111:
        result = ir.MinUnsigned(old, s);
        break;
    default:
        result = s;
        break;
    }

    const IR::U32 status = ExclusiveMem(address, bytes, acctype, AtomicTruncate(*this, result, bytes));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rt, ok, old);
    return AtomicFinish(*this, status);
}

bool TranslatorVisitor::LDAPR_sz(Imm<2> size, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    const size_t regsize = bytes == 8 ? 64 : 32;
    const IR::UAny data = Mem(AtomicAddress(*this, Rn), bytes, IR::AccType::ORDERED);
    X(regsize, Rt, ZeroExtend(data, regsize));
    return true;
}

bool TranslatorVisitor::CompareAndSwap(Imm<2> size, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(L, o0);

    const IR::U64 old = AtomicLoadZext(*this, address, bytes, acctype);
    const IR::U64 expected = AtomicZeroExtendReg(*this, Rs, bytes);
    const IR::U64 desired = AtomicZeroExtendReg(*this, Rt, bytes);
    const IR::U64 equal = AtomicMaskIfZero(*this, ir.Eor(old, expected));
    const IR::U64 result = AtomicSelect(*this, equal, desired, old);

    const IR::U32 status = ExclusiveMem(address, bytes, acctype, AtomicTruncate(*this, result, bytes));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rs, ok, old);
    return AtomicFinish(*this, status);
}

bool TranslatorVisitor::CompareAndSwapPair(bool sz, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    if ((static_cast<size_t>(Rs) & 1) != 0 || (static_cast<size_t>(Rt) & 1) != 0) {
        return UnallocatedEncoding();
    }
    const Reg Rs2 = AtomicNextReg(Rs);
    const Reg Rt2 = AtomicNextReg(Rt);
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(L, o0);

    if (!sz) {
        const IR::U64 old = ExclusiveMem(address, 8, acctype);
        const IR::U64 expected = ir.Pack2x32To1x64(X(32, Rs), X(32, Rs2));
        const IR::U64 desired = ir.Pack2x32To1x64(X(32, Rt), X(32, Rt2));
        const IR::U64 equal = AtomicMaskIfZero(*this, ir.Eor(old, expected));
        const IR::U32 status = ExclusiveMem(address, 8, acctype, AtomicSelect(*this, equal, desired, old));
        const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
        AtomicCommitReg(*this, Rs, ok, ir.ZeroExtendToLong(ir.LeastSignificantWord(old)));
        AtomicCommitReg(*this, Rs2, ok, ir.LogicalShiftRight(old, ir.Imm8(32)));
        return AtomicFinish(*this, status);
    }

    const IR::U128 old = ExclusiveMem(address, 16, acctype);
    const IR::U64 old_lo = ir.VectorGetElement(64, old, 0);
    const IR::U64 old_hi = ir.VectorGetElement(64, old, 1);
    const IR::U64 diff = ir.Or(ir.Eor(old_lo, X(64, Rs)), ir.Eor(old_hi, X(64, Rs2)));
    const IR::U64 equal = AtomicMaskIfZero(*this, diff);
    const IR::U64 new_lo = AtomicSelect(*this, equal, X(64, Rt), old_lo);
    const IR::U64 new_hi = AtomicSelect(*this, equal, X(64, Rt2), old_hi);
    const IR::U32 status = ExclusiveMem(address, 16, acctype, ir.Pack2x64To1x128(new_lo, new_hi));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rs, ok, old_lo);
    AtomicCommitReg(*this, Rs2, ok, old_hi);
    return AtomicFinish(*this, status);
}

}  // namespace Dynarmic::A64
