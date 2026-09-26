// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright (c) Ryujinx Team and Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// AArch64 SIMD and floating-point cases ported from Ryujinx's CPU tests
// (src/Ryujinx.Tests/Cpu, MIT License, Copyright (c) Ryujinx Team and Contributors): their
// opcode lists and value sets, checked against Unicorn. Where Ryujinx draws the rounding mode,
// FZ or DN at random, every combination runs here.

#include <algorithm>
#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/decoder/a64.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/frontend/A64/translate/impl/impl.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/tests/A64/testenv.h"
#include "dynarmic/tests/A64/unicorn_harness.h"
#include "dynarmic/tests/fuzz_util.h"
#include "dynarmic/tests/rand_int.h"
#include "dynarmic/tests/unicorn_emu/a64_unicorn.h"

namespace {

using namespace UnicornFuzz;

// clang-format off
// Generated from Ryujinx a82350bb (src/Ryujinx.Tests/Cpu) by tests/ryujinx_extract.py.
// CpuTestSimd.cs _1H_F_
constexpr std::array<u64, 16> k1H_F{{
    0x000000000000FBFF,  // -Max Normal
    0x0000000000008400,  // -Min Normal
    0x00000000000083FF,  // -Max Subnormal
    0x0000000000008001,  // -Min Subnormal
    0x0000000000007BFF,  // +Max Normal
    0x0000000000000400,  // +Min Normal
    0x00000000000003FF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal
    0x0000000000008000,  // -Zero
    0x0000000000000000,  // +Zero
    0x000000000000FC00,  // -Infinity
    0x0000000000007C00,  // +Infinity
    0x000000000000FE00,  // -QNaN (all zeros payload)
    0x000000000000FDFF,  // -SNaN (all ones  payload)
    0x0000000000007E00,  // +QNaN (all zeros payload) (DefaultNaN)
    0x0000000000007DFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimd.cs _1S_F_
constexpr std::array<u64, 16> k1S_F{{
    0x00000000FF7FFFFF,  // -Max Normal    (float.MinValue)
    0x0000000080800000,  // -Min Normal
    0x00000000807FFFFF,  // -Max Subnormal
    0x0000000080000001,  // -Min Subnormal (-float.Epsilon)
    0x000000007F7FFFFF,  // +Max Normal    (float.MaxValue)
    0x0000000000800000,  // +Min Normal
    0x00000000007FFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (float.Epsilon)
    0x0000000080000000,  // -Zero
    0x0000000000000000,  // +Zero
    0x00000000FF800000,  // -Infinity
    0x000000007F800000,  // +Infinity
    0x00000000FFC00000,  // -QNaN (all zeros payload) (float.NaN)
    0x00000000FFBFFFFF,  // -SNaN (all ones  payload)
    0x000000007FC00000,  // +QNaN (all zeros payload) (-float.NaN) (DefaultNaN)
    0x000000007FBFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimd.cs _1S_F_W_
constexpr std::array<u64, 25> k1S_F_W{{
    0x00000000CF000001,  // -2.1474839E9f  (-2147483904)
    0x00000000CF000000,  // -2.14748365E9f (-2147483648)
    0x00000000CEFFFFFF,  // -2.14748352E9f (-2147483520)
    0x000000004F000001,  // 2.1474839E9f  (2147483904)
    0x000000004F000000,  // 2.14748365E9f (2147483648)
    0x000000004EFFFFFF,  // 2.14748352E9f (2147483520)
    0x000000004F800001,  // 4.2949678E9f  (4294967808)
    0x000000004F800000,  // 4.2949673E9f  (4294967296)
    0x000000004F7FFFFF,  // 4.29496704E9f (4294967040)
    0x00000000FF7FFFFF,  // -Max Normal    (float.MinValue)
    0x0000000080800000,  // -Min Normal
    0x00000000807FFFFF,  // -Max Subnormal
    0x0000000080000001,  // -Min Subnormal (-float.Epsilon)
    0x000000007F7FFFFF,  // +Max Normal    (float.MaxValue)
    0x0000000000800000,  // +Min Normal
    0x00000000007FFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (float.Epsilon)
    0x0000000080000000,  // -Zero
    0x0000000000000000,  // +Zero
    0x00000000FF800000,  // -Infinity
    0x000000007F800000,  // +Infinity
    0x00000000FFC00000,  // -QNaN (all zeros payload) (float.NaN)
    0x00000000FFBFFFFF,  // -SNaN (all ones  payload)
    0x000000007FC00000,  // +QNaN (all zeros payload) (-float.NaN) (DefaultNaN)
    0x000000007FBFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimd.cs _1D_F_
constexpr std::array<u64, 16> k1D_F{{
    0xFFEFFFFFFFFFFFFF,  // -Max Normal    (double.MinValue)
    0x8010000000000000,  // -Min Normal
    0x800FFFFFFFFFFFFF,  // -Max Subnormal
    0x8000000000000001,  // -Min Subnormal (-double.Epsilon)
    0x7FEFFFFFFFFFFFFF,  // +Max Normal    (double.MaxValue)
    0x0010000000000000,  // +Min Normal
    0x000FFFFFFFFFFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (double.Epsilon)
    0x8000000000000000,  // -Zero
    0x0000000000000000,  // +Zero
    0xFFF0000000000000,  // -Infinity
    0x7FF0000000000000,  // +Infinity
    0xFFF8000000000000,  // -QNaN (all zeros payload) (double.NaN)
    0xFFF7FFFFFFFFFFFF,  // -SNaN (all ones  payload)
    0x7FF8000000000000,  // +QNaN (all zeros payload) (-double.NaN) (DefaultNaN)
    0x7FF7FFFFFFFFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimd.cs _1D_F_X_
constexpr std::array<u64, 25> k1D_F_X{{
    0xC3E0000000000001,  // -9.2233720368547780E18d (-9223372036854778000)
    0xC3E0000000000000,  // -9.2233720368547760E18d (-9223372036854776000)
    0xC3DFFFFFFFFFFFFF,  // -9.2233720368547750E18d (-9223372036854775000)
    0x43E0000000000001,  // 9.2233720368547780E18d (9223372036854778000)
    0x43E0000000000000,  // 9.2233720368547760E18d (9223372036854776000)
    0x43DFFFFFFFFFFFFF,  // 9.2233720368547750E18d (9223372036854775000)
    0x43F0000000000001,  // 1.8446744073709556e19d (18446744073709556000)
    0x43F0000000000000,  // 1.8446744073709552E19d (18446744073709552000)
    0x43EFFFFFFFFFFFFF,  // 1.8446744073709550e19d (18446744073709550000)
    0xFFEFFFFFFFFFFFFF,  // -Max Normal    (double.MinValue)
    0x8010000000000000,  // -Min Normal
    0x800FFFFFFFFFFFFF,  // -Max Subnormal
    0x8000000000000001,  // -Min Subnormal (-double.Epsilon)
    0x7FEFFFFFFFFFFFFF,  // +Max Normal    (double.MaxValue)
    0x0010000000000000,  // +Min Normal
    0x000FFFFFFFFFFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (double.Epsilon)
    0x8000000000000000,  // -Zero
    0x0000000000000000,  // +Zero
    0xFFF0000000000000,  // -Infinity
    0x7FF0000000000000,  // +Infinity
    0xFFF8000000000000,  // -QNaN (all zeros payload) (double.NaN)
    0xFFF7FFFFFFFFFFFF,  // -SNaN (all ones  payload)
    0x7FF8000000000000,  // +QNaN (all zeros payload) (-double.NaN) (DefaultNaN)
    0x7FF7FFFFFFFFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimdCvt.cs _1S_F_WX_
constexpr std::array<u64, 34> k1S_F_WX{{
    0x00000000CF000001,  // -2.1474839E9f  (-2147483904)
    0x00000000CF000000,  // -2.14748365E9f (-2147483648)
    0x00000000CEFFFFFF,  // -2.14748352E9f (-2147483520)
    0x000000004F000001,  // 2.1474839E9f  (2147483904)
    0x000000004F000000,  // 2.14748365E9f (2147483648)
    0x000000004EFFFFFF,  // 2.14748352E9f (2147483520)
    0x00000000DF000001,  // -9.223373E18f  (-9223373136366403584)
    0x00000000DF000000,  // -9.223372E18f  (-9223372036854775808)
    0x00000000DEFFFFFF,  // -9.2233715E18f (-9223371487098961920)
    0x000000005F000001,  // 9.223373E18f  (9223373136366403584)
    0x000000005F000000,  // 9.223372E18f  (9223372036854775808)
    0x000000005EFFFFFF,  // 9.2233715E18f (9223371487098961920)
    0x000000004F800001,  // 4.2949678E9f  (4294967808)
    0x000000004F800000,  // 4.2949673E9f  (4294967296)
    0x000000004F7FFFFF,  // 4.29496704E9f (4294967040)
    0x000000005F800001,  // 1.8446746E19f (18446746272732807168)
    0x000000005F800000,  // 1.8446744E19f (18446744073709551616)
    0x000000005F7FFFFF,  // 1.8446743E19f (18446742974197923840)
    0x00000000FF7FFFFF,  // -Max Normal    (float.MinValue)
    0x0000000080800000,  // -Min Normal
    0x00000000807FFFFF,  // -Max Subnormal
    0x0000000080000001,  // -Min Subnormal (-float.Epsilon)
    0x000000007F7FFFFF,  // +Max Normal    (float.MaxValue)
    0x0000000000800000,  // +Min Normal
    0x00000000007FFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (float.Epsilon)
    0x0000000080000000,  // -Zero
    0x0000000000000000,  // +Zero
    0x00000000FF800000,  // -Infinity
    0x000000007F800000,  // +Infinity
    0x00000000FFC00000,  // -QNaN (all zeros payload) (float.NaN)
    0x00000000FFBFFFFF,  // -SNaN (all ones  payload)
    0x000000007FC00000,  // +QNaN (all zeros payload) (-float.NaN) (DefaultNaN)
    0x000000007FBFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimdCvt.cs _1D_F_WX_
constexpr std::array<u64, 34> k1D_F_WX{{
    0xC1E0000000200000,  // -2147483649.0000000d (-2147483649)
    0xC1E0000000000000,  // -2147483648.0000000d (-2147483648)
    0xC1DFFFFFFFC00000,  // -2147483647.0000000d (-2147483647)
    0x41E0000000200000,  // 2147483649.0000000d (2147483649)
    0x41E0000000000000,  // 2147483648.0000000d (2147483648)
    0x41DFFFFFFFC00000,  // 2147483647.0000000d (2147483647)
    0xC3E0000000000001,  // -9.2233720368547780E18d (-9223372036854778000)
    0xC3E0000000000000,  // -9.2233720368547760E18d (-9223372036854776000)
    0xC3DFFFFFFFFFFFFF,  // -9.2233720368547750E18d (-9223372036854775000)
    0x43E0000000000001,  // 9.2233720368547780E18d (9223372036854778000)
    0x43E0000000000000,  // 9.2233720368547760E18d (9223372036854776000)
    0x43DFFFFFFFFFFFFF,  // 9.2233720368547750E18d (9223372036854775000)
    0x41F0000000100000,  // 4294967297.0000000d (4294967297)
    0x41F0000000000000,  // 4294967296.0000000d (4294967296)
    0x41EFFFFFFFE00000,  // 4294967295.0000000d (4294967295)
    0x43F0000000000001,  // 1.8446744073709556e19d (18446744073709556000)
    0x43F0000000000000,  // 1.8446744073709552E19d (18446744073709552000)
    0x43EFFFFFFFFFFFFF,  // 1.8446744073709550e19d (18446744073709550000)
    0xFFEFFFFFFFFFFFFF,  // -Max Normal    (double.MinValue)
    0x8010000000000000,  // -Min Normal
    0x800FFFFFFFFFFFFF,  // -Max Subnormal
    0x8000000000000001,  // -Min Subnormal (-double.Epsilon)
    0x7FEFFFFFFFFFFFFF,  // +Max Normal    (double.MaxValue)
    0x0010000000000000,  // +Min Normal
    0x000FFFFFFFFFFFFF,  // +Max Subnormal
    0x0000000000000001,  // +Min Subnormal (double.Epsilon)
    0x8000000000000000,  // -Zero
    0x0000000000000000,  // +Zero
    0xFFF0000000000000,  // -Infinity
    0x7FF0000000000000,  // +Infinity
    0xFFF8000000000000,  // -QNaN (all zeros payload) (double.NaN)
    0xFFF7FFFFFFFFFFFF,  // -SNaN (all ones  payload)
    0x7FF8000000000000,  // +QNaN (all zeros payload) (-double.NaN) (DefaultNaN)
    0x7FF7FFFFFFFFFFFF,  // +SNaN (all ones  payload)
}};
// CpuTestSimdCvt.cs _W_
constexpr std::array<u32, 4> kW{{
    0x00000000,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
}};
// CpuTestSimdCvt.cs _X_
constexpr std::array<u64, 4> kX{{
    0x0000000000000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
}};
// CpuTestSimd.cs _1S_
constexpr std::array<u64, 4> k1S{{
    0x0000000000000000,
    0x000000007FFFFFFF,
    0x0000000080000000,
    0x00000000FFFFFFFF,
}};
// CpuTestSimd.cs _2S_
constexpr std::array<u64, 4> k2S{{
    0x0000000000000000,
    0x7FFFFFFF7FFFFFFF,
    0x8000000080000000,
    0xFFFFFFFFFFFFFFFF,
}};
// CpuTestSimd.cs _1D_
constexpr std::array<u64, 4> k1D{{
    0x0000000000000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
}};
// CpuTestSimd.cs _F_Abs_Neg_Recpx_Sqrt_S_S_
constexpr std::array<u32, 4> kF_Abs_Neg_Recpx_Sqrt_S_S{{
    0x1E20C020,  // FABS   S0, S1
    0x1E214020,  // FNEG   S0, S1
    0x5EA1F820,  // FRECPX S0, S1
    0x1E21C020,  // FSQRT  S0, S1
}};
// CpuTestSimd.cs _F_Abs_Neg_Recpx_Sqrt_S_D_
constexpr std::array<u32, 4> kF_Abs_Neg_Recpx_Sqrt_S_D{{
    0x1E60C020,  // FABS   D0, D1
    0x1E614020,  // FNEG   D0, D1
    0x5EE1F820,  // FRECPX D0, D1
    0x1E61C020,  // FSQRT  D0, D1
}};
// CpuTestSimd.cs _F_Abs_Neg_Sqrt_V_2S_4S_
constexpr std::array<u32, 3> kF_Abs_Neg_Sqrt_V_2S_4S{{
    0x0EA0F800,  // FABS  V0.2S, V0.2S
    0x2EA0F800,  // FNEG  V0.2S, V0.2S
    0x2EA1F800,  // FSQRT V0.2S, V0.2S
}};
// CpuTestSimd.cs _F_Abs_Neg_Sqrt_V_2D_
constexpr std::array<u32, 3> kF_Abs_Neg_Sqrt_V_2D{{
    0x4EE0F800,  // FABS  V0.2D, V0.2D
    0x6EE0F800,  // FNEG  V0.2D, V0.2D
    0x6EE1F800,  // FSQRT V0.2D, V0.2D
}};
// CpuTestSimd.cs _F_Add_Max_Min_Nm_P_S_2SS_
constexpr std::array<u32, 5> kF_Add_Max_Min_Nm_P_S_2SS{{
    0x7E30D820,  // FADDP   S0, V1.2S
    0x7E30C820,  // FMAXNMP S0, V1.2S
    0x7E30F820,  // FMAXP   S0, V1.2S
    0x7EB0C820,  // FMINNMP S0, V1.2S
    0x7EB0F820,  // FMINP   S0, V1.2S
}};
// CpuTestSimd.cs _F_Add_Max_Min_Nm_P_S_2DD_
constexpr std::array<u32, 5> kF_Add_Max_Min_Nm_P_S_2DD{{
    0x7E70D820,  // FADDP   D0, V1.2D
    0x7E70C820,  // FMAXNMP D0, V1.2D
    0x7E70F820,  // FMAXP   D0, V1.2D
    0x7EF0C820,  // FMINNMP D0, V1.2D
    0x7EF0F820,  // FMINP   D0, V1.2D
}};
// CpuTestSimd.cs _F_Cm_EqGeGtLeLt_S_S_
constexpr std::array<u32, 5> kF_Cm_EqGeGtLeLt_S_S{{
    0x5EA0D820,  // FCMEQ S0, S1, #0.0
    0x7EA0C820,  // FCMGE S0, S1, #0.0
    0x5EA0C820,  // FCMGT S0, S1, #0.0
    0x7EA0D820,  // FCMLE S0, S1, #0.0
    0x5EA0E820,  // FCMLT S0, S1, #0.0
}};
// CpuTestSimd.cs _F_Cm_EqGeGtLeLt_S_D_
constexpr std::array<u32, 5> kF_Cm_EqGeGtLeLt_S_D{{
    0x5EE0D820,  // FCMEQ D0, D1, #0.0
    0x7EE0C820,  // FCMGE D0, D1, #0.0
    0x5EE0C820,  // FCMGT D0, D1, #0.0
    0x7EE0D820,  // FCMLE D0, D1, #0.0
    0x5EE0E820,  // FCMLT D0, D1, #0.0
}};
// CpuTestSimd.cs _F_Cm_EqGeGtLeLt_V_2S_4S_
constexpr std::array<u32, 5> kF_Cm_EqGeGtLeLt_V_2S_4S{{
    0x0EA0D800,  // FCMEQ V0.2S, V0.2S, #0.0
    0x2EA0C800,  // FCMGE V0.2S, V0.2S, #0.0
    0x0EA0C800,  // FCMGT V0.2S, V0.2S, #0.0
    0x2EA0D800,  // FCMLE V0.2S, V0.2S, #0.0
    0x0EA0E800,  // FCMLT V0.2S, V0.2S, #0.0
}};
// CpuTestSimd.cs _F_Cm_EqGeGtLeLt_V_2D_
constexpr std::array<u32, 5> kF_Cm_EqGeGtLeLt_V_2D{{
    0x4EE0D800,  // FCMEQ V0.2D, V0.2D, #0.0
    0x6EE0C800,  // FCMGE V0.2D, V0.2D, #0.0
    0x4EE0C800,  // FCMGT V0.2D, V0.2D, #0.0
    0x6EE0D800,  // FCMLE V0.2D, V0.2D, #0.0
    0x4EE0E800,  // FCMLT V0.2D, V0.2D, #0.0
}};
// CpuTestSimd.cs _F_Cmp_Cmpe_S_S_ (compare with zero)
constexpr std::array<u32, 2> kF_Cmp_Cmpe_Zero_S_S{{
    0x1E202028,  // FCMP  S1, #0.0
    0x1E202038,  // FCMPE S1, #0.0
}};
// CpuTestSimd.cs _F_Cmp_Cmpe_S_D_ (compare with zero)
constexpr std::array<u32, 2> kF_Cmp_Cmpe_Zero_S_D{{
    0x1E602028,  // FCMP  D1, #0.0
    0x1E602038,  // FCMPE D1, #0.0
}};
// CpuTestSimd.cs _F_Cvt_S_SD_
constexpr std::array<u32, 1> kF_Cvt_S_SD{{
    0x1E22C020,  // FCVT D0, S1
}};
// CpuTestSimd.cs _F_Cvt_S_DS_
constexpr std::array<u32, 1> kF_Cvt_S_DS{{
    0x1E624020,  // FCVT S0, D1
}};
// CpuTestSimd.cs _F_Cvt_S_SH_
constexpr std::array<u32, 1> kF_Cvt_S_SH{{
    0x1E23C020,  // FCVT H0, S1
}};
// CpuTestSimd.cs _F_Cvt_S_DH_
constexpr std::array<u32, 1> kF_Cvt_S_DH{{
    0x1E63C020,  // FCVT H0, D1
}};
// CpuTestSimd.cs _F_Cvt_S_HS_
constexpr std::array<u32, 1> kF_Cvt_S_HS{{
    0x1EE24020,  // FCVT S0, H1
}};
// CpuTestSimd.cs _F_Cvt_S_HD_
constexpr std::array<u32, 1> kF_Cvt_S_HD{{
    0x1EE2C020,  // FCVT D0, H1
}};
// CpuTestSimd.cs _F_Cvt_ANZ_SU_S_S_
constexpr std::array<u32, 6> kF_Cvt_ANZ_SU_S_S{{
    0x5E21C820,  // FCVTAS S0, S1
    0x7E21C820,  // FCVTAU S0, S1
    0x5E21A820,  // FCVTNS S0, S1
    0x7E21A820,  // FCVTNU S0, S1
    0x5EA1B820,  // FCVTZS S0, S1
    0x7EA1B820,  // FCVTZU S0, S1
}};
// CpuTestSimd.cs _F_Cvt_ANZ_SU_S_D_
constexpr std::array<u32, 6> kF_Cvt_ANZ_SU_S_D{{
    0x5E61C820,  // FCVTAS D0, D1
    0x7E61C820,  // FCVTAU D0, D1
    0x5E61A820,  // FCVTNS D0, D1
    0x7E61A820,  // FCVTNU D0, D1
    0x5EE1B820,  // FCVTZS D0, D1
    0x7EE1B820,  // FCVTZU D0, D1
}};
// CpuTestSimd.cs _F_Cvt_ANZ_SU_V_2S_4S_
constexpr std::array<u32, 7> kF_Cvt_ANZ_SU_V_2S_4S{{
    0x0E21C800,  // FCVTAS V0.2S, V0.2S
    0x2E21C800,  // FCVTAU V0.2S, V0.2S
    0x0E21B800,  // FCVTMS V0.2S, V0.2S
    0x0E21A800,  // FCVTNS V0.2S, V0.2S
    0x2E21A800,  // FCVTNU V0.2S, V0.2S
    0x0EA1B800,  // FCVTZS V0.2S, V0.2S
    0x2EA1B800,  // FCVTZU V0.2S, V0.2S
}};
// CpuTestSimd.cs _F_Cvt_ANZ_SU_V_2D_
constexpr std::array<u32, 7> kF_Cvt_ANZ_SU_V_2D{{
    0x4E61C800,  // FCVTAS V0.2D, V0.2D
    0x6E61C800,  // FCVTAU V0.2D, V0.2D
    0x4E61B800,  // FCVTMS V0.2D, V0.2D
    0x4E61A800,  // FCVTNS V0.2D, V0.2D
    0x6E61A800,  // FCVTNU V0.2D, V0.2D
    0x4EE1B800,  // FCVTZS V0.2D, V0.2D
    0x6EE1B800,  // FCVTZU V0.2D, V0.2D
}};
// CpuTestSimd.cs _F_Cvtl_V_4H4S_8H4S_
constexpr std::array<u32, 1> kF_Cvtl_V_4H4S_8H4S{{
    0x0E217800,  // FCVTL V0.4S, V0.4H
}};
// CpuTestSimd.cs _F_Cvtl_V_2S2D_4S2D_
constexpr std::array<u32, 1> kF_Cvtl_V_2S2D_4S2D{{
    0x0E617800,  // FCVTL V0.2D, V0.2S
}};
// CpuTestSimd.cs _F_Cvtn_V_4S4H_4S8H_
constexpr std::array<u32, 1> kF_Cvtn_V_4S4H_4S8H{{
    0x0E216800,  // FCVTN V0.4H, V0.4S
}};
// CpuTestSimd.cs _F_Cvtn_V_2D2S_2D4S_
constexpr std::array<u32, 1> kF_Cvtn_V_2D2S_2D4S{{
    0x0E616800,  // FCVTN V0.2S, V0.2D
}};
// CpuTestSimd.cs _F_Max_Min_Nm_V_V_4SS_
constexpr std::array<u32, 4> kF_Max_Min_Nm_V_V_4SS{{
    0x6E30C800,  // FMAXNMV S0, V0.4S
    0x6E30F800,  // FMAXV   S0, V0.4S
    0x6EB0C800,  // FMINNMV S0, V0.4S
    0x6EB0F800,  // FMINV   S0, V0.4S
}};
// CpuTestSimd.cs _F_Recpe_Rsqrte_S_S_
constexpr std::array<u32, 2> kF_Recpe_Rsqrte_S_S{{
    0x5EA1D820,  // FRECPE  S0, S1
    0x7EA1D820,  // FRSQRTE S0, S1
}};
// CpuTestSimd.cs _F_Recpe_Rsqrte_S_D_
constexpr std::array<u32, 2> kF_Recpe_Rsqrte_S_D{{
    0x5EE1D820,  // FRECPE  D0, D1
    0x7EE1D820,  // FRSQRTE D0, D1
}};
// CpuTestSimd.cs _F_Recpe_Rsqrte_V_2S_4S_
constexpr std::array<u32, 2> kF_Recpe_Rsqrte_V_2S_4S{{
    0x0EA1D800,  // FRECPE  V0.2S, V0.2S
    0x2EA1D800,  // FRSQRTE V0.2S, V0.2S
}};
// CpuTestSimd.cs _F_Recpe_Rsqrte_V_2D_
constexpr std::array<u32, 2> kF_Recpe_Rsqrte_V_2D{{
    0x4EE1D800,  // FRECPE  V0.2D, V0.2D
    0x6EE1D800,  // FRSQRTE V0.2D, V0.2D
}};
// CpuTestSimd.cs _F_Rint_AMNPZ_S_S_
constexpr std::array<u32, 5> kF_Rint_AMNPZ_S_S{{
    0x1E264020,  // FRINTA S0, S1
    0x1E254020,  // FRINTM S0, S1
    0x1E244020,  // FRINTN S0, S1
    0x1E24C020,  // FRINTP S0, S1
    0x1E25C020,  // FRINTZ S0, S1
}};
// CpuTestSimd.cs _F_Rint_AMNPZ_S_D_
constexpr std::array<u32, 5> kF_Rint_AMNPZ_S_D{{
    0x1E664020,  // FRINTA D0, D1
    0x1E654020,  // FRINTM D0, D1
    0x1E644020,  // FRINTN D0, D1
    0x1E64C020,  // FRINTP D0, D1
    0x1E65C020,  // FRINTZ D0, D1
}};
// CpuTestSimd.cs _F_Rint_AMNPZ_V_2S_4S_
constexpr std::array<u32, 5> kF_Rint_AMNPZ_V_2S_4S{{
    0x2E218800,  // FRINTA V0.2S, V0.2S
    0x0E219800,  // FRINTM V0.2S, V0.2S
    0x0E218800,  // FRINTN V0.2S, V0.2S
    0x0EA18800,  // FRINTP V0.2S, V0.2S
    0x0EA19800,  // FRINTZ V0.2S, V0.2S
}};
// CpuTestSimd.cs _F_Rint_AMNPZ_V_2D_
constexpr std::array<u32, 5> kF_Rint_AMNPZ_V_2D{{
    0x6E618800,  // FRINTA V0.2D, V0.2D
    0x4E619800,  // FRINTM V0.2D, V0.2D
    0x4E618800,  // FRINTN V0.2D, V0.2D
    0x4EE18800,  // FRINTP V0.2D, V0.2D
    0x4EE19800,  // FRINTZ V0.2D, V0.2D
}};
// CpuTestSimd.cs _F_Rint_IX_S_S_
constexpr std::array<u32, 2> kF_Rint_IX_S_S{{
    0x1E27C020,  // FRINTI S0, S1
    0x1E274020,  // FRINTX S0, S1
}};
// CpuTestSimd.cs _F_Rint_IX_S_D_
constexpr std::array<u32, 2> kF_Rint_IX_S_D{{
    0x1E67C020,  // FRINTI D0, D1
    0x1E674020,  // FRINTX D0, D1
}};
// CpuTestSimd.cs _F_Rint_IX_V_2S_4S_
constexpr std::array<u32, 2> kF_Rint_IX_V_2S_4S{{
    0x2EA19800,  // FRINTI V0.2S, V0.2S
    0x2E219800,  // FRINTX V0.2S, V0.2S
}};
// CpuTestSimd.cs _F_Rint_IX_V_2D_
constexpr std::array<u32, 2> kF_Rint_IX_V_2D{{
    0x6EE19800,  // FRINTI V0.2D, V0.2D
    0x6E619800,  // FRINTX V0.2D, V0.2D
}};
// CpuTestSimd.cs _SU_Cvt_F_S_S_
constexpr std::array<u32, 2> kSU_Cvt_F_S_S{{
    0x5E21D820,  // SCVTF S0, S1
    0x7E21D820,  // UCVTF S0, S1
}};
// CpuTestSimd.cs _SU_Cvt_F_S_D_
constexpr std::array<u32, 2> kSU_Cvt_F_S_D{{
    0x5E61D820,  // SCVTF D0, D1
    0x7E61D820,  // UCVTF D0, D1
}};
// CpuTestSimd.cs _SU_Cvt_F_V_2S_4S_
constexpr std::array<u32, 2> kSU_Cvt_F_V_2S_4S{{
    0x0E21D800,  // SCVTF V0.2S, V0.2S
    0x2E21D800,  // UCVTF V0.2S, V0.2S
}};
// CpuTestSimd.cs _SU_Cvt_F_V_2D_
constexpr std::array<u32, 2> kSU_Cvt_F_V_2D{{
    0x4E61D800,  // SCVTF V0.2D, V0.2D
    0x6E61D800,  // UCVTF V0.2D, V0.2D
}};
// CpuTestSimdReg.cs _F_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_S_
constexpr std::array<u32, 7> kF_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_S{{
    0x7EA2D420,  // FABD  S0, S1, S2
    0x1E222820,  // FADD  S0, S1, S2
    0x1E221820,  // FDIV  S0, S1, S2
    0x1E220820,  // FMUL  S0, S1, S2
    0x5E22DC20,  // FMULX S0, S1, S2
    0x1E228820,  // FNMUL S0, S1, S2
    0x1E223820,  // FSUB  S0, S1, S2
}};
// CpuTestSimdReg.cs _F_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_D_
constexpr std::array<u32, 7> kF_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_D{{
    0x7EE2D420,  // FABD  D0, D1, D2
    0x1E622820,  // FADD  D0, D1, D2
    0x1E621820,  // FDIV  D0, D1, D2
    0x1E620820,  // FMUL  D0, D1, D2
    0x5E62DC20,  // FMULX D0, D1, D2
    0x1E628820,  // FNMUL D0, D1, D2
    0x1E623820,  // FSUB  D0, D1, D2
}};
// CpuTestSimdReg.cs _F_Abd_Add_Div_Mul_Mulx_Sub_P_V_2S_4S_
constexpr std::array<u32, 7> kF_Abd_Add_Div_Mul_Mulx_Sub_P_V_2S_4S{{
    0x2EA0D400,  // FABD  V0.2S, V0.2S, V0.2S
    0x0E20D400,  // FADD  V0.2S, V0.2S, V0.2S
    0x2E20D400,  // FADDP V0.2S, V0.2S, V0.2S
    0x2E20FC00,  // FDIV  V0.2S, V0.2S, V0.2S
    0x2E20DC00,  // FMUL  V0.2S, V0.2S, V0.2S
    0x0E20DC00,  // FMULX V0.2S, V0.2S, V0.2S
    0x0EA0D400,  // FSUB  V0.2S, V0.2S, V0.2S
}};
// CpuTestSimdReg.cs _F_Abd_Add_Div_Mul_Mulx_Sub_P_V_2D_
constexpr std::array<u32, 7> kF_Abd_Add_Div_Mul_Mulx_Sub_P_V_2D{{
    0x6EE0D400,  // FABD  V0.2D, V0.2D, V0.2D
    0x4E60D400,  // FADD  V0.2D, V0.2D, V0.2D
    0x6E60D400,  // FADDP V0.2D, V0.2D, V0.2D
    0x6E60FC00,  // FDIV  V0.2D, V0.2D, V0.2D
    0x6E60DC00,  // FMUL  V0.2D, V0.2D, V0.2D
    0x4E60DC00,  // FMULX V0.2D, V0.2D, V0.2D
    0x4EE0D400,  // FSUB  V0.2D, V0.2D, V0.2D
}};
// CpuTestSimdReg.cs _F_AcCm_EqGeGt_S_S_
constexpr std::array<u32, 5> kF_AcCm_EqGeGt_S_S{{
    0x7E22EC20,  // FACGE S0, S1, S2
    0x7EA2EC20,  // FACGT S0, S1, S2
    0x5E22E420,  // FCMEQ S0, S1, S2
    0x7E22E420,  // FCMGE S0, S1, S2
    0x7EA2E420,  // FCMGT S0, S1, S2
}};
// CpuTestSimdReg.cs _F_AcCm_EqGeGt_S_D_
constexpr std::array<u32, 5> kF_AcCm_EqGeGt_S_D{{
    0x7E62EC20,  // FACGE D0, D1, D2
    0x7EE2EC20,  // FACGT D0, D1, D2
    0x5E62E420,  // FCMEQ D0, D1, D2
    0x7E62E420,  // FCMGE D0, D1, D2
    0x7EE2E420,  // FCMGT D0, D1, D2
}};
// CpuTestSimdReg.cs _F_AcCm_EqGeGt_V_2S_4S_
constexpr std::array<u32, 5> kF_AcCm_EqGeGt_V_2S_4S{{
    0x2E20EC00,  // FACGE V0.2S, V0.2S, V0.2S
    0x2EA0EC00,  // FACGT V0.2S, V0.2S, V0.2S
    0x0E20E400,  // FCMEQ V0.2S, V0.2S, V0.2S
    0x2E20E400,  // FCMGE V0.2S, V0.2S, V0.2S
    0x2EA0E400,  // FCMGT V0.2S, V0.2S, V0.2S
}};
// CpuTestSimdReg.cs _F_AcCm_EqGeGt_V_2D_
constexpr std::array<u32, 5> kF_AcCm_EqGeGt_V_2D{{
    0x6E60EC00,  // FACGE V0.2D, V0.2D, V0.2D
    0x6EE0EC00,  // FACGT V0.2D, V0.2D, V0.2D
    0x4E60E400,  // FCMEQ V0.2D, V0.2D, V0.2D
    0x6E60E400,  // FCMGE V0.2D, V0.2D, V0.2D
    0x6EE0E400,  // FCMGT V0.2D, V0.2D, V0.2D
}};
// CpuTestSimdReg.cs _F_Cmp_Cmpe_S_S_
constexpr std::array<u32, 2> kF_Cmp_Cmpe_S_S{{
    0x1E222020,  // FCMP  S1, S2
    0x1E222030,  // FCMPE S1, S2
}};
// CpuTestSimdReg.cs _F_Cmp_Cmpe_S_D_
constexpr std::array<u32, 2> kF_Cmp_Cmpe_S_D{{
    0x1E622020,  // FCMP  D1, D2
    0x1E622030,  // FCMPE D1, D2
}};
// CpuTestSimdReg.cs _F_Madd_Msub_Nmadd_Nmsub_S_S_
constexpr std::array<u32, 4> kF_Madd_Msub_Nmadd_Nmsub_S_S{{
    0x1F020C20,  // FMADD  S0, S1, S2, S3
    0x1F028C20,  // FMSUB  S0, S1, S2, S3
    0x1F220C20,  // FNMADD S0, S1, S2, S3
    0x1F228C20,  // FNMSUB S0, S1, S2, S3
}};
// CpuTestSimdReg.cs _F_Madd_Msub_Nmadd_Nmsub_S_D_
constexpr std::array<u32, 4> kF_Madd_Msub_Nmadd_Nmsub_S_D{{
    0x1F420C20,  // FMADD  D0, D1, D2, D3
    0x1F428C20,  // FMSUB  D0, D1, D2, D3
    0x1F620C20,  // FNMADD D0, D1, D2, D3
    0x1F628C20,  // FNMSUB D0, D1, D2, D3
}};
// CpuTestSimdReg.cs _F_Max_Min_Nm_S_S_
constexpr std::array<u32, 4> kF_Max_Min_Nm_S_S{{
    0x1E224820,  // FMAX   S0, S1, S2
    0x1E226820,  // FMAXNM S0, S1, S2
    0x1E225820,  // FMIN   S0, S1, S2
    0x1E227820,  // FMINNM S0, S1, S2
}};
// CpuTestSimdReg.cs _F_Max_Min_Nm_S_D_
constexpr std::array<u32, 4> kF_Max_Min_Nm_S_D{{
    0x1E624820,  // FMAX   D0, D1, D2
    0x1E626820,  // FMAXNM D0, D1, D2
    0x1E625820,  // FMIN   D0, D1, D2
    0x1E627820,  // FMINNM D0, D1, D2
}};
// CpuTestSimdReg.cs _F_Max_Min_Nm_P_V_2S_4S_
constexpr std::array<u32, 8> kF_Max_Min_Nm_P_V_2S_4S{{
    0x0E20F400,  // FMAX    V0.2S, V0.2S, V0.2S
    0x0E20C400,  // FMAXNM  V0.2S, V0.2S, V0.2S
    0x2E20C400,  // FMAXNMP V0.2S, V0.2S, V0.2S
    0x2E20F400,  // FMAXP   V0.2S, V0.2S, V0.2S
    0x0EA0F400,  // FMIN    V0.2S, V0.2S, V0.2S
    0x0EA0C400,  // FMINNM  V0.2S, V0.2S, V0.2S
    0x2EA0C400,  // FMINNMP V0.2S, V0.2S, V0.2S
    0x2EA0F400,  // FMINP   V0.2S, V0.2S, V0.2S
}};
// CpuTestSimdReg.cs _F_Max_Min_Nm_P_V_2D_
constexpr std::array<u32, 8> kF_Max_Min_Nm_P_V_2D{{
    0x4E60F400,  // FMAX    V0.2D, V0.2D, V0.2D
    0x4E60C400,  // FMAXNM  V0.2D, V0.2D, V0.2D
    0x6E60C400,  // FMAXNMP V0.2D, V0.2D, V0.2D
    0x6E60F400,  // FMAXP   V0.2D, V0.2D, V0.2D
    0x4EE0F400,  // FMIN    V0.2D, V0.2D, V0.2D
    0x4EE0C400,  // FMINNM  V0.2D, V0.2D, V0.2D
    0x6EE0C400,  // FMINNMP V0.2D, V0.2D, V0.2D
    0x6EE0F400,  // FMINP   V0.2D, V0.2D, V0.2D
}};
// CpuTestSimdReg.cs _F_Mla_Mls_V_2S_4S_
constexpr std::array<u32, 2> kF_Mla_Mls_V_2S_4S{{
    0x0E20CC00,  // FMLA V0.2S, V0.2S, V0.2S
    0x0EA0CC00,  // FMLS V0.2S, V0.2S, V0.2S
}};
// CpuTestSimdReg.cs _F_Mla_Mls_V_2D_
constexpr std::array<u32, 2> kF_Mla_Mls_V_2D{{
    0x4E60CC00,  // FMLA V0.2D, V0.2D, V0.2D
    0x4EE0CC00,  // FMLS V0.2D, V0.2D, V0.2D
}};
// CpuTestSimdReg.cs _F_Recps_Rsqrts_S_S_
constexpr std::array<u32, 2> kF_Recps_Rsqrts_S_S{{
    0x5E22FC20,  // FRECPS  S0, S1, S2
    0x5EA2FC20,  // FRSQRTS S0, S1, S2
}};
// CpuTestSimdReg.cs _F_Recps_Rsqrts_S_D_
constexpr std::array<u32, 2> kF_Recps_Rsqrts_S_D{{
    0x5E62FC20,  // FRECPS  D0, D1, D2
    0x5EE2FC20,  // FRSQRTS D0, D1, D2
}};
// CpuTestSimdReg.cs _F_Recps_Rsqrts_V_2S_4S_
constexpr std::array<u32, 2> kF_Recps_Rsqrts_V_2S_4S{{
    0x0E20FC00,  // FRECPS  V0.2S, V0.2S, V0.2S
    0x0EA0FC00,  // FRSQRTS V0.2S, V0.2S, V0.2S
}};
// CpuTestSimdReg.cs _F_Recps_Rsqrts_V_2D_
constexpr std::array<u32, 2> kF_Recps_Rsqrts_V_2D{{
    0x4E60FC00,  // FRECPS  V0.2D, V0.2D, V0.2D
    0x4EE0FC00,  // FRSQRTS V0.2D, V0.2D, V0.2D
}};
// CpuTestSimdCvt.cs _F_Cvt_AMPZ_SU_Gp_SW_
constexpr std::array<u32, 9> kF_Cvt_AMPZ_SU_Gp_SW{{
    0x1E240000,  // FCVTAS W0, S0
    0x1E250000,  // FCVTAU W0, S0
    0x1E300000,  // FCVTMS W0, S0
    0x1E310000,  // FCVTMU W0, S0
    0x1E200000,  // FCVTNS W0, S0
    0x1E280000,  // FCVTPS W0, S0
    0x1E290000,  // FCVTPU W0, S0
    0x1E380000,  // FCVTZS W0, S0
    0x1E390000,  // FCVTZU W0, S0
}};
// CpuTestSimdCvt.cs _F_Cvt_AMPZ_SU_Gp_SX_
constexpr std::array<u32, 9> kF_Cvt_AMPZ_SU_Gp_SX{{
    0x9E240000,  // FCVTAS X0, S0
    0x9E250000,  // FCVTAU X0, S0
    0x9E300000,  // FCVTMS X0, S0
    0x9E310000,  // FCVTMU X0, S0
    0x9E200000,  // FCVTNS X0, S0
    0x9E280000,  // FCVTPS X0, S0
    0x9E290000,  // FCVTPU X0, S0
    0x9E380000,  // FCVTZS X0, S0
    0x9E390000,  // FCVTZU X0, S0
}};
// CpuTestSimdCvt.cs _F_Cvt_AMPZ_SU_Gp_DW_
constexpr std::array<u32, 9> kF_Cvt_AMPZ_SU_Gp_DW{{
    0x1E640000,  // FCVTAS W0, D0
    0x1E650000,  // FCVTAU W0, D0
    0x1E700000,  // FCVTMS W0, D0
    0x1E710000,  // FCVTMU W0, D0
    0x1E600000,  // FCVTNS W0, D0
    0x1E680000,  // FCVTPS W0, D0
    0x1E690000,  // FCVTPU W0, D0
    0x1E780000,  // FCVTZS W0, D0
    0x1E790000,  // FCVTZU W0, D0
}};
// CpuTestSimdCvt.cs _F_Cvt_AMPZ_SU_Gp_DX_
constexpr std::array<u32, 9> kF_Cvt_AMPZ_SU_Gp_DX{{
    0x9E640000,  // FCVTAS X0, D0
    0x9E650000,  // FCVTAU X0, D0
    0x9E700000,  // FCVTMS X0, D0
    0x9E710000,  // FCVTMU X0, D0
    0x9E600000,  // FCVTNS X0, D0
    0x9E680000,  // FCVTPS X0, D0
    0x9E690000,  // FCVTPU X0, D0
    0x9E780000,  // FCVTZS X0, D0
    0x9E790000,  // FCVTZU X0, D0
}};
// CpuTestSimdCvt.cs _F_Cvt_Z_SU_Gp_Fixed_SW_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_Gp_Fixed_SW{{
    0x1E188000,  // FCVTZS W0, S0, #32
    0x1E198000,  // FCVTZU W0, S0, #32
}};
// CpuTestSimdCvt.cs _F_Cvt_Z_SU_Gp_Fixed_SX_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_Gp_Fixed_SX{{
    0x9E180000,  // FCVTZS X0, S0, #64
    0x9E190000,  // FCVTZU X0, S0, #64
}};
// CpuTestSimdCvt.cs _F_Cvt_Z_SU_Gp_Fixed_DW_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_Gp_Fixed_DW{{
    0x1E588000,  // FCVTZS W0, D0, #32
    0x1E598000,  // FCVTZU W0, D0, #32
}};
// CpuTestSimdCvt.cs _F_Cvt_Z_SU_Gp_Fixed_DX_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_Gp_Fixed_DX{{
    0x9E580000,  // FCVTZS X0, D0, #64
    0x9E590000,  // FCVTZU X0, D0, #64
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_WS_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_WS{{
    0x1E220000,  // SCVTF S0, W0
    0x1E230000,  // UCVTF S0, W0
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_WD_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_WD{{
    0x1E620000,  // SCVTF D0, W0
    0x1E630000,  // UCVTF D0, W0
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_XS_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_XS{{
    0x9E220000,  // SCVTF S0, X0
    0x9E230000,  // UCVTF S0, X0
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_XD_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_XD{{
    0x9E620000,  // SCVTF D0, X0
    0x9E630000,  // UCVTF D0, X0
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_Fixed_WS_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_Fixed_WS{{
    0x1E028000,  // SCVTF S0, W0, #32
    0x1E038000,  // UCVTF S0, W0, #32
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_Fixed_WD_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_Fixed_WD{{
    0x1E428000,  // SCVTF D0, W0, #32
    0x1E438000,  // UCVTF D0, W0, #32
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_Fixed_XS_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_Fixed_XS{{
    0x9E020000,  // SCVTF S0, X0, #64
    0x9E030000,  // UCVTF S0, X0, #64
}};
// CpuTestSimdCvt.cs _SU_Cvt_F_Gp_Fixed_XD_
constexpr std::array<u32, 2> kSU_Cvt_F_Gp_Fixed_XD{{
    0x9E420000,  // SCVTF D0, X0, #64
    0x9E430000,  // UCVTF D0, X0, #64
}};
// CpuTestSimdShImm.cs _F_Cvt_Z_SU_V_Fixed_2S_4S_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_V_Fixed_2S_4S{{
    0x0F20FC00,  // FCVTZS V0.2S, V0.2S, #32
    0x2F20FC00,  // FCVTZU V0.2S, V0.2S, #32
}};
// CpuTestSimdShImm.cs _F_Cvt_Z_SU_V_Fixed_2D_
constexpr std::array<u32, 2> kF_Cvt_Z_SU_V_Fixed_2D{{
    0x4F40FC00,  // FCVTZS V0.2D, V0.2D, #64
    0x6F40FC00,  // FCVTZU V0.2D, V0.2D, #64
}};
// CpuTestSimdShImm.cs _SU_Cvt_F_S_Fixed_S_
constexpr std::array<u32, 2> kSU_Cvt_F_S_Fixed_S{{
    0x5F20E420,  // SCVTF S0, S1, #32
    0x7F20E420,  // UCVTF S0, S1, #32
}};
// CpuTestSimdShImm.cs _SU_Cvt_F_S_Fixed_D_
constexpr std::array<u32, 2> kSU_Cvt_F_S_Fixed_D{{
    0x5F40E420,  // SCVTF D0, D1, #64
    0x7F40E420,  // UCVTF D0, D1, #64
}};
// CpuTestSimdShImm.cs _SU_Cvt_F_V_Fixed_2S_4S_
constexpr std::array<u32, 2> kSU_Cvt_F_V_Fixed_2S_4S{{
    0x0F20E400,  // SCVTF V0.2S, V0.2S, #32
    0x2F20E400,  // UCVTF V0.2S, V0.2S, #32
}};
// CpuTestSimdShImm.cs _SU_Cvt_F_V_Fixed_2D_
constexpr std::array<u32, 2> kSU_Cvt_F_V_Fixed_2D{{
    0x4F40E400,  // SCVTF V0.2D, V0.2D, #64
    0x6F40E400,  // UCVTF V0.2D, V0.2D, #64
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_S_HB_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_S_HB{{
    0x5F089C00,  // SQRSHRN  B0, H0, #8
    0x7F089C00,  // UQRSHRN  B0, H0, #8
    0x7F088C00,  // SQRSHRUN B0, H0, #8
    0x5F089400,  // SQSHRN   B0, H0, #8
    0x7F089400,  // UQSHRN   B0, H0, #8
    0x7F088400,  // SQSHRUN  B0, H0, #8
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_S_SH_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_S_SH{{
    0x5F109C00,  // SQRSHRN  H0, S0, #16
    0x7F109C00,  // UQRSHRN  H0, S0, #16
    0x7F108C00,  // SQRSHRUN H0, S0, #16
    0x5F109400,  // SQSHRN   H0, S0, #16
    0x7F109400,  // UQSHRN   H0, S0, #16
    0x7F108400,  // SQSHRUN  H0, S0, #16
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_S_DS_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_S_DS{{
    0x5F209C00,  // SQRSHRN  S0, D0, #32
    0x7F209C00,  // UQRSHRN  S0, D0, #32
    0x7F208C00,  // SQRSHRUN S0, D0, #32
    0x5F209400,  // SQSHRN   S0, D0, #32
    0x7F209400,  // UQSHRN   S0, D0, #32
    0x7F208400,  // SQSHRUN  S0, D0, #32
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_V_8H8B_8H16B_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_V_8H8B_8H16B{{
    0x0F089C00,  // SQRSHRN  V0.8B, V0.8H, #8
    0x2F089C00,  // UQRSHRN  V0.8B, V0.8H, #8
    0x2F088C00,  // SQRSHRUN V0.8B, V0.8H, #8
    0x0F089400,  // SQSHRN   V0.8B, V0.8H, #8
    0x2F089400,  // UQSHRN   V0.8B, V0.8H, #8
    0x2F088400,  // SQSHRUN  V0.8B, V0.8H, #8
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_V_4S4H_4S8H_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_V_4S4H_4S8H{{
    0x0F109C00,  // SQRSHRN  V0.4H, V0.4S, #16
    0x2F109C00,  // UQRSHRN  V0.4H, V0.4S, #16
    0x2F108C00,  // SQRSHRUN V0.4H, V0.4S, #16
    0x0F109400,  // SQSHRN   V0.4H, V0.4S, #16
    0x2F109400,  // UQSHRN   V0.4H, V0.4S, #16
    0x2F108400,  // SQSHRUN  V0.4H, V0.4S, #16
}};
// CpuTestSimdShImm.cs _ShrImmSaturatingNarrow_V_2D2S_2D4S_
constexpr std::array<u32, 6> kShrImmSaturatingNarrow_V_2D2S_2D4S{{
    0x0F209C00,  // SQRSHRN  V0.2S, V0.2D, #32
    0x2F209C00,  // UQRSHRN  V0.2S, V0.2D, #32
    0x2F208C00,  // SQRSHRUN V0.2S, V0.2D, #32
    0x0F209400,  // SQSHRN   V0.2S, V0.2D, #32
    0x2F209400,  // UQSHRN   V0.2S, V0.2D, #32
    0x2F208400,  // SQSHRUN  V0.2S, V0.2D, #32
}};
// CpuTestSimdShImm.cs _1H_
constexpr std::array<u64, 4> k1H{{
    0x0000000000000000,
    0x0000000000007FFF,
    0x0000000000008000,
    0x000000000000FFFF,
}};
// CpuTestSimdShImm.cs _4H_
constexpr std::array<u64, 4> k4H{{
    0x0000000000000000,
    0x7FFF7FFF7FFF7FFF,
    0x8000800080008000,
    0xFFFFFFFFFFFFFFFF,
}};
// CpuTestSimdShImm.cs _8B_
constexpr std::array<u64, 4> k8B{{
    0x0000000000000000,
    0x7F7F7F7F7F7F7F7F,
    0x8080808080808080,
    0xFFFFFFFFFFFFFFFF,
}};
// CpuTestSimdRegElemF.cs _F_Mla_Mls_Se_S_
constexpr std::array<u32, 2> kF_Mla_Mls_Se_S{{
    0x5F821020,  // FMLA S0, S1, V2.S[0]
    0x5F825020,  // FMLS S0, S1, V2.S[0]
}};
// CpuTestSimdRegElemF.cs _F_Mla_Mls_Se_D_
constexpr std::array<u32, 2> kF_Mla_Mls_Se_D{{
    0x5FC21020,  // FMLA D0, D1, V2.D[0]
    0x5FC25020,  // FMLS D0, D1, V2.D[0]
}};
// CpuTestSimdRegElemF.cs _F_Mul_Mulx_Se_S_
constexpr std::array<u32, 2> kF_Mul_Mulx_Se_S{{
    0x5F829020,  // FMUL  S0, S1, V2.S[0]
    0x7F829020,  // FMULX S0, S1, V2.S[0]
}};
// CpuTestSimdRegElemF.cs _F_Mul_Mulx_Se_D_
constexpr std::array<u32, 2> kF_Mul_Mulx_Se_D{{
    0x5FC29020,  // FMUL  D0, D1, V2.D[0]
    0x7FC29020,  // FMULX D0, D1, V2.D[0]
}};
// CpuTestSimdFcond.cs _F_Ccmp_Ccmpe_S_S_
constexpr std::array<u32, 2> kF_Ccmp_Ccmpe_S_S{{
    0x1E220420,  // FCCMP  S1, S2, #0, EQ
    0x1E220430,  // FCCMPE S1, S2, #0, EQ
}};
// CpuTestSimdFcond.cs _F_Ccmp_Ccmpe_S_D_
constexpr std::array<u32, 2> kF_Ccmp_Ccmpe_S_D{{
    0x1E620420,  // FCCMP  D1, D2, #0, EQ
    0x1E620430,  // FCCMPE D1, D2, #0, EQ
}};
// CpuTestSimdFcond.cs _F_Csel_S_S_
constexpr std::array<u32, 1> kF_Csel_S_S{{
    0x1E220C20,  // FCSEL S0, S1, S2, EQ
}};
// CpuTestSimdFcond.cs _F_Csel_S_D_
constexpr std::array<u32, 1> kF_Csel_S_D{{
    0x1E620C20,  // FCSEL D0, D1, D2, EQ
}};
// clang-format on

// Ryujinx's GenNormal*/GenSubnormal* (CpuTest.cs): random finite values of one class.
u64 GenNormalH() {
    u64 rnd;
    do
        rnd = RandInt<u16>(0, 0xFFFF);
    while ((rnd & 0x7C00) == 0 || (~rnd & 0x7C00) == 0);
    return rnd;
}
u64 GenSubnormalH() {
    u64 rnd;
    do
        rnd = RandInt<u16>(0, 0xFFFF);
    while ((rnd & 0x03FF) == 0);
    return rnd & 0x83FF;
}
u64 GenNormalS() {
    u64 rnd;
    do
        rnd = RandInt<u32>(0, 0xFFFFFFFF);
    while ((rnd & 0x7F800000) == 0 || (~rnd & 0x7F800000) == 0);
    return rnd;
}
u64 GenSubnormalS() {
    u64 rnd;
    do
        rnd = RandInt<u32>(0, 0xFFFFFFFF);
    while ((rnd & 0x007FFFFF) == 0);
    return rnd & 0x807FFFFF;
}
u64 GenNormalD() {
    u64 rnd;
    do
        rnd = RandInt<u64>(0, ~u64(0));
    while ((rnd & 0x7FF0000000000000) == 0 || (~rnd & 0x7FF0000000000000) == 0);
    return rnd;
}
u64 GenSubnormalD() {
    u64 rnd;
    do
        rnd = RandInt<u64>(0, ~u64(0));
    while ((rnd & 0x000FFFFFFFFFFFFF) == 0);
    return rnd & 0x800FFFFFFFFFFFFF;
}

// Ryujinx appends RndCnt (2) random normals and subnormals to each fixed value set.
template<typename Array>
std::vector<u64> WithRandom(const Array& fixed, u64 (*normal)(), u64 (*subnormal)()) {
    std::vector<u64> values(fixed.begin(), fixed.end());
    for (size_t i = 0; i < 2; ++i) {
        values.push_back(normal());
        values.push_back(subnormal());
    }
    return values;
}

// The <2S>, <4H> sets repeat one element over the doubleword, as Ryujinx's _2S_F_ and _4H_F_ do.
std::vector<u64> Repeat(const std::vector<u64>& values, size_t esize) {
    std::vector<u64> out;
    for (const u64 value : values) {
        const u64 element = value & ((u64(1) << esize) - 1);
        u64 repeated = 0;
        for (size_t i = 0; i < 64; i += esize)
            repeated |= element << i;
        out.push_back(repeated);
    }
    return out;
}

std::vector<u64> ValuesH() { return WithRandom(k1H_F, GenNormalH, GenSubnormalH); }
std::vector<u64> ValuesS() { return WithRandom(k1S_F, GenNormalS, GenSubnormalS); }
std::vector<u64> ValuesD() { return WithRandom(k1D_F, GenNormalD, GenSubnormalD); }

constexpr u32 kFpcrRModeShift = 22;
constexpr u32 kFpcrFz16 = 1 << 19;
constexpr u32 kFpcrFz = 1 << 24;
constexpr u32 kFpcrDn = 1 << 25;
constexpr u32 kFpcrAhp = 1 << 26;

// Every rounding mode with FZ and DN on and off; half precision also flips AHP and FZ16.
std::vector<u32> AllFpcr(bool half = false) {
    std::vector<u32> out;
    for (u32 rmode = 0; rmode < 4; ++rmode) {
        for (u32 bits = 0; bits < (half ? 16u : 4u); ++bits) {
            u32 fpcr = rmode << kFpcrRModeShift;
            fpcr |= (bits & 1) ? kFpcrFz : 0;
            fpcr |= (bits & 2) ? kFpcrDn : 0;
            fpcr |= (bits & 4) ? kFpcrAhp : 0;
            fpcr |= (bits & 8) ? kFpcrFz16 : 0;
            out.push_back(fpcr);
        }
    }
    return out;
}

// For the three-operand and the long sweeps: FZ and DN both ways, round to nearest.
std::vector<u32> FzDnFpcr() {
    return {0, kFpcrFz, kFpcrDn, kFpcrFz | kFpcrDn};
}

class Runner {
public:
    Runner()
            : conf{GetUserConfig(jit_env)}, jit{conf}, uni{uni_env} {}

    // Runs one instruction over random state that `setup` then overwrites, once per FPCR value.
    template<typename Setup>
    void Run(u32 opcode, const std::vector<u32>& fpcrs, Setup&& setup) {
        if (RaisesException(opcode)) {
            // The test env aborts on ExceptionRaised, so these are reported instead of run.
            fmt::print("=== dynarmic raises an exception for {:08x} {}\n", opcode, Name(opcode));
            if (!IsKnownUnimplemented(opcode))
                ++mismatches;
            return;
        }
        for (const u32 fpcr : fpcrs) {
            A64FuzzCase c;
            std::generate(c.regs.begin(), c.regs.end(), [] { return RandInt<u64>(0, ~u64(0)); });
            std::generate(c.vecs.begin(), c.vecs.end(), RandomVector);
            c.start_address = RandInt<u64>(0, 0x10'0000'0000) * 4;
            c.initial_sp = RandInt<u64>(0x30'0000'0000, 0x40'0000'0000) * 4;
            c.instructions = {opcode};
            c.pstate = RandInt<u32>(0, 0xF) << 28;
            c.fpcr = fpcr;
            setup(c);
            mismatches += !RunFuzzCase(jit, uni, jit_env, uni_env, conf, c, runs++, FuzzReport::Brief);
        }
    }

    size_t runs = 0;
    size_t mismatches = 0;

private:
    static std::string_view Name(u32 opcode) {
        const auto name = Dynarmic::A64::GetName<Dynarmic::A64::TranslatorVisitor>(opcode);
        return name ? *name : std::string_view{"<unallocated>"};
    }

    // Unallocated in dynarmic's decoder, or translated to an exception.
    // dynarmic has no scalar SQRSHRN, UQRSHRN or SQRSHRUN; the vector forms and the
    // non-rounding scalar forms are implemented.
    static bool IsKnownUnimplemented(u32 opcode) {
        return (opcode & 0xFF80FC00) == 0x5F009C00      // SQRSHRN (scalar)
               || (opcode & 0xFF80FC00) == 0x7F009C00   // UQRSHRN (scalar)
               || (opcode & 0xFF80FC00) == 0x7F008C00;  // SQRSHRUN (scalar)
    }

    static bool RaisesException(u32 opcode) {
        if (!Dynarmic::A64::GetName<Dynarmic::A64::TranslatorVisitor>(opcode))
            return true;
        const Dynarmic::A64::LocationDescriptor location{0, {}};
        Dynarmic::IR::Block block{location};
        Dynarmic::A64::TranslateSingleInstruction(block, location, opcode);
        return std::any_of(block.instructions.begin(), block.instructions.end(), [](const auto& inst) {
            return inst.GetOpcode() == Dynarmic::IR::Opcode::A64ExceptionRaised;
        });
    }

    A64TestEnv jit_env{};
    A64TestEnv uni_env{};
    Dynarmic::A64::UserConfig conf;
    Dynarmic::A64::Jit jit;
    A64Unicorn uni;
};

u64 Garbage() {
    return RandInt<u64>(0, ~u64(0));
}

// Scalar unary, registers in the opcode (Rd = 0, Rn = 1): v0 = garbage, v1 = <a, 0>.
template<typename Opcodes>
void ScalarUnary(Runner& r, const Opcodes& opcodes, const std::vector<u64>& values, const std::vector<u32>& fpcrs) {
    for (const u32 opcode : opcodes) {
        for (const u64 a : values) {
            r.Run(opcode, fpcrs, [&](A64FuzzCase& c) {
                c.vecs[0] = {Garbage(), Garbage()};
                c.vecs[1] = {a, 0};
            });
        }
    }
}

// Scalar binary (Rd = 0, Rn = 1, Rm = 2): every pair of values.
template<typename Opcodes>
void ScalarBinary(Runner& r, const Opcodes& opcodes, const std::vector<u64>& values, const std::vector<u32>& fpcrs) {
    for (const u32 opcode : opcodes) {
        for (const u64 a : values) {
            for (const u64 b : values) {
                r.Run(opcode, fpcrs, [&](A64FuzzCase& c) {
                    c.vecs[0] = {Garbage(), Garbage()};
                    c.vecs[1] = {a, 0};
                    c.vecs[2] = {b, 0};
                });
            }
        }
    }
}

// Vector unary: Rd = 0, Rn = 1 or 0 (in place), Q both ways when `has_q`; v1 = <a, a * q>.
template<typename Opcodes>
void VectorUnary(Runner& r, const Opcodes& opcodes, const std::vector<u64>& values, bool has_q, const std::vector<u32>& fpcrs) {
    for (u32 opcode : opcodes) {
        for (const u32 q : {0u, 1u}) {
            if (q && !has_q)
                continue;
            for (const u32 rn : {1u, 0u}) {
                const u32 inst = opcode | (rn << 5) | (q << 30);
                for (const u64 a : values) {
                    r.Run(inst, fpcrs, [&](A64FuzzCase& c) {
                        const u64 upper = has_q ? a * q : a;
                        if (rn == 0) {
                            c.vecs[0] = {a, upper};
                        } else {
                            c.vecs[0] = {Garbage(), Garbage()};
                            c.vecs[1] = {a, upper};
                        }
                    });
                }
            }
        }
    }
}

// Vector binary: Rd = 0, Rn = 1, Rm = 2, Q both ways when `has_q`; every pair of values.
template<typename Opcodes>
void VectorBinary(Runner& r, const Opcodes& opcodes, const std::vector<u64>& values, bool has_q, const std::vector<u32>& fpcrs) {
    for (u32 opcode : opcodes) {
        for (const u32 q : {0u, 1u}) {
            if (q && !has_q)
                continue;
            const u32 inst = opcode | (2 << 16) | (1 << 5) | (q << 30);
            for (const u64 a : values) {
                for (const u64 b : values) {
                    r.Run(inst, fpcrs, [&](A64FuzzCase& c) {
                        c.vecs[0] = {Garbage(), Garbage()};
                        c.vecs[1] = {a, has_q ? a * q : a};
                        c.vecs[2] = {b, has_q ? b * q : b};
                    });
                }
            }
        }
    }
}

// Scalar with a condition field (FCCMP, FCSEL): every condition, every pair of values.
template<typename Opcodes>
void ScalarConditional(Runner& r, const Opcodes& opcodes, const std::vector<u64>& values, bool has_nzcv_imm) {
    for (const u32 opcode : opcodes) {
        for (u32 cond = 0; cond < 16; ++cond) {
            const u32 inst = opcode | (cond << 12) | (has_nzcv_imm ? RandInt<u32>(0, 15) : 0);
            for (const u64 a : values) {
                for (const u64 b : values) {
                    r.Run(inst, {0}, [&](A64FuzzCase& c) {
                        c.vecs[0] = {Garbage(), Garbage()};
                        c.vecs[1] = {a, 0};
                        c.vecs[2] = {b, 0};
                    });
                }
            }
        }
    }
}

}  // namespace

TEST_CASE("Ryujinx A64: FABS FNEG FRECPX FSQRT", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Abs_Neg_Recpx_Sqrt_S_S, ValuesS(), AllFpcr());
    ScalarUnary(r, kF_Abs_Neg_Recpx_Sqrt_S_D, ValuesD(), AllFpcr());
    VectorUnary(r, kF_Abs_Neg_Sqrt_V_2S_4S, Repeat(ValuesS(), 32), true, AllFpcr());
    VectorUnary(r, kF_Abs_Neg_Sqrt_V_2D, ValuesD(), false, AllFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FRINT", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Rint_AMNPZ_S_S, ValuesS(), AllFpcr());
    ScalarUnary(r, kF_Rint_AMNPZ_S_D, ValuesD(), AllFpcr());
    ScalarUnary(r, kF_Rint_IX_S_S, ValuesS(), AllFpcr());
    ScalarUnary(r, kF_Rint_IX_S_D, ValuesD(), AllFpcr());
    VectorUnary(r, kF_Rint_AMNPZ_V_2S_4S, Repeat(ValuesS(), 32), true, AllFpcr());
    VectorUnary(r, kF_Rint_AMNPZ_V_2D, ValuesD(), false, AllFpcr());
    VectorUnary(r, kF_Rint_IX_V_2S_4S, Repeat(ValuesS(), 32), true, AllFpcr());
    VectorUnary(r, kF_Rint_IX_V_2D, ValuesD(), false, AllFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FRECPE FRSQRTE", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Recpe_Rsqrte_S_S, ValuesS(), AllFpcr());
    ScalarUnary(r, kF_Recpe_Rsqrte_S_D, ValuesD(), AllFpcr());
    VectorUnary(r, kF_Recpe_Rsqrte_V_2S_4S, Repeat(ValuesS(), 32), true, AllFpcr());
    VectorUnary(r, kF_Recpe_Rsqrte_V_2D, ValuesD(), false, AllFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCM against zero", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Cm_EqGeGtLeLt_S_S, ValuesS(), AllFpcr());
    ScalarUnary(r, kF_Cm_EqGeGtLeLt_S_D, ValuesD(), AllFpcr());
    VectorUnary(r, kF_Cm_EqGeGtLeLt_V_2S_4S, Repeat(ValuesS(), 32), true, AllFpcr());
    VectorUnary(r, kF_Cm_EqGeGtLeLt_V_2D, ValuesD(), false, AllFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCMP FCMPE", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Cmp_Cmpe_Zero_S_S, ValuesS(), FzDnFpcr());
    ScalarUnary(r, kF_Cmp_Cmpe_Zero_S_D, ValuesD(), FzDnFpcr());
    ScalarBinary(r, kF_Cmp_Cmpe_S_S, ValuesS(), FzDnFpcr());
    ScalarBinary(r, kF_Cmp_Cmpe_S_D, ValuesD(), FzDnFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCVT between precisions", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarUnary(r, kF_Cvt_S_SD, ValuesS(), AllFpcr(true));
    ScalarUnary(r, kF_Cvt_S_DS, ValuesD(), AllFpcr(true));
    ScalarUnary(r, kF_Cvt_S_SH, ValuesS(), AllFpcr(true));
    ScalarUnary(r, kF_Cvt_S_DH, ValuesD(), AllFpcr(true));
    ScalarUnary(r, kF_Cvt_S_HS, ValuesH(), AllFpcr(true));
    ScalarUnary(r, kF_Cvt_S_HD, ValuesH(), AllFpcr(true));
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCVTL FCVTN", "[a64][unicorn][ryujinx]") {
    Runner r;
    // FCVTL reads the lower half (2S/4H) or, with Q, the upper half (4S/8H) of Vn.
    for (const auto& [opcodes, values] : {std::pair{kF_Cvtl_V_4H4S_8H4S, Repeat(ValuesH(), 16)}, std::pair{kF_Cvtl_V_2S2D_4S2D, Repeat(ValuesS(), 32)}}) {
        for (const u32 q : {0u, 1u}) {
            for (const u64 a : values) {
                r.Run(opcodes[0] | (1 << 5) | (q << 30), AllFpcr(true), [&](A64FuzzCase& c) {
                    c.vecs[0] = {Garbage(), Garbage()};
                    c.vecs[1] = {q == 0 ? a : 0, q == 1 ? a : 0};
                });
            }
        }
    }
    // FCVTN writes the lower half, or with Q the upper half keeping the lower one.
    for (const auto& [opcodes, values] : {std::pair{kF_Cvtn_V_4S4H_4S8H, Repeat(ValuesS(), 32)}, std::pair{kF_Cvtn_V_2D2S_2D4S, ValuesD()}}) {
        for (const u32 q : {0u, 1u}) {
            for (const u64 a : values) {
                r.Run(opcodes[0] | (1 << 5) | (q << 30), AllFpcr(true), [&](A64FuzzCase& c) {
                    const u64 z = Garbage();
                    c.vecs[0] = {z, a};
                    c.vecs[1] = {a, z};
                });
            }
        }
    }
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCVT to integer in SIMD registers", "[a64][unicorn][ryujinx]") {
    Runner r;
    const auto values_s = WithRandom(k1S_F_W, GenNormalS, GenSubnormalS);
    const auto values_d = WithRandom(k1D_F_X, GenNormalD, GenSubnormalD);
    ScalarUnary(r, kF_Cvt_ANZ_SU_S_S, values_s, AllFpcr());
    ScalarUnary(r, kF_Cvt_ANZ_SU_S_D, values_d, AllFpcr());
    VectorUnary(r, kF_Cvt_ANZ_SU_V_2S_4S, Repeat(values_s, 32), true, AllFpcr());
    VectorUnary(r, kF_Cvt_ANZ_SU_V_2D, values_d, false, AllFpcr());
    // Fixed point: FCVTZS/FCVTZU (vector, fixed-point) with fbits 1 and the element size.
    for (const u32 fbits : {1u, 32u}) {
        for (u32 opcode : kF_Cvt_Z_SU_V_Fixed_2S_4S)
            VectorUnary(r, std::array{opcode | (((64 - fbits) & 0x7F) << 16)}, Repeat(values_s, 32), true, FzDnFpcr());
    }
    for (const u32 fbits : {1u, 64u}) {
        for (u32 opcode : kF_Cvt_Z_SU_V_Fixed_2D)
            VectorUnary(r, std::array{opcode | (((128 - fbits) & 0x7F) << 16)}, values_d, false, FzDnFpcr());
    }
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCVT to general registers", "[a64][unicorn][ryujinx]") {
    Runner r;
    const auto values_s = WithRandom(k1S_F_WX, GenNormalS, GenSubnormalS);
    const auto values_d = WithRandom(k1D_F_WX, GenNormalD, GenSubnormalD);
    // Rd is X0 or XZR, Rn is V1; FPCR.RMode does not apply, FZ still does.
    const auto run = [&](const auto& opcodes, const std::vector<u64>& values, u32 extra) {
        for (const u32 opcode : opcodes) {
            for (const u32 rd : {0u, 31u}) {
                for (const u64 a : values) {
                    r.Run(opcode | extra | (1 << 5) | rd, FzDnFpcr(), [&](A64FuzzCase& c) {
                        c.vecs[1] = {a, 0};
                    });
                }
            }
        }
    };
    run(kF_Cvt_AMPZ_SU_Gp_SW, values_s, 0);
    run(kF_Cvt_AMPZ_SU_Gp_SX, values_s, 0);
    run(kF_Cvt_AMPZ_SU_Gp_DW, values_d, 0);
    run(kF_Cvt_AMPZ_SU_Gp_DX, values_d, 0);
    for (const u32 fbits : {1u, 32u}) {
        run(kF_Cvt_Z_SU_Gp_Fixed_SW, values_s, ((64 - fbits) & 0x3F) << 10);
        run(kF_Cvt_Z_SU_Gp_Fixed_DW, values_d, ((64 - fbits) & 0x3F) << 10);
    }
    for (const u32 fbits : {1u, 64u}) {
        run(kF_Cvt_Z_SU_Gp_Fixed_SX, values_s, ((64 - fbits) & 0x3F) << 10);
        run(kF_Cvt_Z_SU_Gp_Fixed_DX, values_d, ((64 - fbits) & 0x3F) << 10);
    }
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: SCVTF UCVTF", "[a64][unicorn][ryujinx]") {
    Runner r;
    const auto with_random = [](const auto& fixed, u64 max) {
        std::vector<u64> values(fixed.begin(), fixed.end());
        for (size_t i = 0; i < 4; ++i)
            values.push_back(RandInt<u64>(0, max));
        return values;
    };
    const auto w = with_random(kW, 0xFFFFFFFF);
    const auto x = with_random(kX, ~u64(0));
    // From general registers: Rd = V0, Rn = X1 or XZR, every rounding mode.
    const auto from_gpr = [&](const auto& opcodes, const std::vector<u64>& values, u32 extra) {
        for (const u32 opcode : opcodes) {
            for (const u32 rn : {1u, 31u}) {
                for (const u64 a : values) {
                    r.Run(opcode | extra | (rn << 5), AllFpcr(), [&](A64FuzzCase& c) {
                        c.regs[1] = a;
                        c.vecs[0] = {Garbage(), Garbage()};
                    });
                }
            }
        }
    };
    from_gpr(kSU_Cvt_F_Gp_WS, w, 0);
    from_gpr(kSU_Cvt_F_Gp_WD, w, 0);
    from_gpr(kSU_Cvt_F_Gp_XS, x, 0);
    from_gpr(kSU_Cvt_F_Gp_XD, x, 0);
    for (const u32 fbits : {1u, 32u}) {
        from_gpr(kSU_Cvt_F_Gp_Fixed_WS, w, ((64 - fbits) & 0x3F) << 10);
        from_gpr(kSU_Cvt_F_Gp_Fixed_WD, w, ((64 - fbits) & 0x3F) << 10);
    }
    for (const u32 fbits : {1u, 64u}) {
        from_gpr(kSU_Cvt_F_Gp_Fixed_XS, x, ((64 - fbits) & 0x3F) << 10);
        from_gpr(kSU_Cvt_F_Gp_Fixed_XD, x, ((64 - fbits) & 0x3F) << 10);
    }
    // From SIMD registers.
    const auto s = with_random(k1S, 0xFFFFFFFF);
    const auto d = with_random(k1D, ~u64(0));
    ScalarUnary(r, kSU_Cvt_F_S_S, s, AllFpcr());
    ScalarUnary(r, kSU_Cvt_F_S_D, d, AllFpcr());
    VectorUnary(r, kSU_Cvt_F_V_2S_4S, Repeat(s, 32), true, AllFpcr());
    VectorUnary(r, kSU_Cvt_F_V_2D, d, false, AllFpcr());
    for (const u32 fbits : {1u, 32u}) {
        for (const u32 opcode : kSU_Cvt_F_S_Fixed_S)
            ScalarUnary(r, std::array{opcode | (((64 - fbits) & 0x7F) << 16)}, s, AllFpcr());
        for (const u32 opcode : kSU_Cvt_F_V_Fixed_2S_4S)
            VectorUnary(r, std::array{opcode | (((64 - fbits) & 0x7F) << 16)}, Repeat(s, 32), true, AllFpcr());
    }
    for (const u32 fbits : {1u, 64u}) {
        for (const u32 opcode : kSU_Cvt_F_S_Fixed_D)
            ScalarUnary(r, std::array{opcode | (((128 - fbits) & 0x7F) << 16)}, d, AllFpcr());
        for (const u32 opcode : kSU_Cvt_F_V_Fixed_2D)
            VectorUnary(r, std::array{opcode | (((128 - fbits) & 0x7F) << 16)}, d, false, AllFpcr());
    }
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FABD FADD FDIV FMUL FMULX FNMUL FSUB", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarBinary(r, kF_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_S, ValuesS(), AllFpcr());
    ScalarBinary(r, kF_Abd_Add_Div_Mul_Mulx_Nmul_Sub_S_D, ValuesD(), AllFpcr());
    VectorBinary(r, kF_Abd_Add_Div_Mul_Mulx_Sub_P_V_2S_4S, Repeat(ValuesS(), 32), true, FzDnFpcr());
    VectorBinary(r, kF_Abd_Add_Div_Mul_Mulx_Sub_P_V_2D, ValuesD(), false, FzDnFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FMAX FMIN FMAXNM FMINNM", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarBinary(r, kF_Max_Min_Nm_S_S, ValuesS(), FzDnFpcr());
    ScalarBinary(r, kF_Max_Min_Nm_S_D, ValuesD(), FzDnFpcr());
    VectorBinary(r, kF_Max_Min_Nm_P_V_2S_4S, Repeat(ValuesS(), 32), true, FzDnFpcr());
    VectorBinary(r, kF_Max_Min_Nm_P_V_2D, ValuesD(), false, FzDnFpcr());
    // Pairwise scalar: the pair is the doubleword of Vn (2S) or its two doublewords (2D).
    for (const u32 opcode : kF_Add_Max_Min_Nm_P_S_2SS) {
        for (const u64 a : Repeat(ValuesS(), 32)) {
            r.Run(opcode, AllFpcr(), [&](A64FuzzCase& c) {
                c.vecs[0] = {Garbage(), Garbage()};
                c.vecs[1] = {a, Garbage()};
            });
        }
    }
    const auto values_d = ValuesD();
    for (const u32 opcode : kF_Add_Max_Min_Nm_P_S_2DD) {
        for (const u64 a0 : values_d) {
            for (const u64 a1 : values_d) {
                r.Run(opcode, FzDnFpcr(), [&](A64FuzzCase& c) {
                    c.vecs[0] = {Garbage(), Garbage()};
                    c.vecs[1] = {a0, a1};
                });
            }
        }
    }
    // Across lanes (4S): FMAXNMV FMAXV FMINNMV FMINV.
    const auto values_s = Repeat(ValuesS(), 32);
    for (const u32 opcode : kF_Max_Min_Nm_V_V_4SS) {
        for (const u64 a : values_s) {
            for (const u64 b : values_s) {
                r.Run(opcode | (1 << 5), FzDnFpcr(), [&](A64FuzzCase& c) {
                    c.vecs[0] = {Garbage(), Garbage()};
                    c.vecs[1] = {a, b};
                });
            }
        }
    }
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FACGE FACGT FCMEQ FCMGE FCMGT", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarBinary(r, kF_AcCm_EqGeGt_S_S, ValuesS(), FzDnFpcr());
    ScalarBinary(r, kF_AcCm_EqGeGt_S_D, ValuesD(), FzDnFpcr());
    VectorBinary(r, kF_AcCm_EqGeGt_V_2S_4S, Repeat(ValuesS(), 32), true, FzDnFpcr());
    VectorBinary(r, kF_AcCm_EqGeGt_V_2D, ValuesD(), false, FzDnFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FRECPS FRSQRTS", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarBinary(r, kF_Recps_Rsqrts_S_S, ValuesS(), FzDnFpcr());
    ScalarBinary(r, kF_Recps_Rsqrts_S_D, ValuesD(), FzDnFpcr());
    VectorBinary(r, kF_Recps_Rsqrts_V_2S_4S, Repeat(ValuesS(), 32), true, FzDnFpcr());
    VectorBinary(r, kF_Recps_Rsqrts_V_2D, ValuesD(), false, FzDnFpcr());
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FMADD FMSUB FNMADD FNMSUB FMLA FMLS", "[a64][unicorn][ryujinx]") {
    Runner r;
    // Three operands: every triple, FZ and DN both ways.
    for (const auto& [opcodes, values] : {std::pair{kF_Madd_Msub_Nmadd_Nmsub_S_S, ValuesS()}, std::pair{kF_Madd_Msub_Nmadd_Nmsub_S_D, ValuesD()}}) {
        for (const u32 opcode : opcodes) {
            for (const u64 a : values) {
                for (const u64 b : values) {
                    for (const u64 c3 : values) {
                        r.Run(opcode, {RandInt<u32>(0, 3) << 24}, [&](A64FuzzCase& c) {
                            c.vecs[0] = {Garbage(), Garbage()};
                            c.vecs[1] = {a, 0};
                            c.vecs[2] = {b, 0};
                            c.vecs[3] = {c3, 0};
                        });
                    }
                }
            }
        }
    }
    // FMLA/FMLS (vector) accumulate into V0.
    const auto accumulate = [&](const auto& opcodes, const std::vector<u64>& values, bool has_q) {
        for (u32 opcode : opcodes) {
            for (const u32 q : {0u, 1u}) {
                if (q && !has_q)
                    continue;
                const u32 inst = opcode | (2 << 16) | (1 << 5) | (q << 30);
                for (const u64 z : values) {
                    for (const u64 a : values) {
                        const u64 b = values[RandInt<size_t>(0, values.size() - 1)];
                        r.Run(inst, FzDnFpcr(), [&](A64FuzzCase& c) {
                            c.vecs[0] = {z, z};
                            c.vecs[1] = {a, has_q ? a * q : a};
                            c.vecs[2] = {b, has_q ? b * q : b};
                        });
                    }
                }
            }
        }
    };
    accumulate(kF_Mla_Mls_V_2S_4S, Repeat(ValuesS(), 32), true);
    accumulate(kF_Mla_Mls_V_2D, ValuesD(), false);
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FMLA FMLS FMUL FMULX by element", "[a64][unicorn][ryujinx]") {
    Runner r;
    const auto by_element = [&](const auto& opcodes, const std::vector<u64>& values, const std::vector<u64>& pairs, u32 indices) {
        for (const u32 opcode : opcodes) {
            for (u32 index = 0; index < indices; ++index) {
                const u32 h = indices == 4 ? (index >> 1) & 1 : index & 1;
                const u32 l = indices == 4 ? index & 1 : 0;
                const u32 inst = opcode | (l << 21) | (h << 11);
                for (const u64 a : values) {
                    for (const u64 b : pairs) {
                        r.Run(inst, FzDnFpcr(), [&](A64FuzzCase& c) {
                            const u64 z = Garbage();
                            c.vecs[0] = {z, z};
                            c.vecs[1] = {a, 0};
                            c.vecs[2] = {b, b * h};
                        });
                    }
                }
            }
        }
    };
    by_element(kF_Mla_Mls_Se_S, ValuesS(), Repeat(ValuesS(), 32), 4);
    by_element(kF_Mul_Mulx_Se_S, ValuesS(), Repeat(ValuesS(), 32), 4);
    by_element(kF_Mla_Mls_Se_D, ValuesD(), ValuesD(), 2);
    by_element(kF_Mul_Mulx_Se_D, ValuesD(), ValuesD(), 2);
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: FCCMP FCCMPE FCSEL", "[a64][unicorn][ryujinx]") {
    Runner r;
    ScalarConditional(r, kF_Ccmp_Ccmpe_S_S, ValuesS(), true);
    ScalarConditional(r, kF_Ccmp_Ccmpe_S_D, ValuesD(), true);
    ScalarConditional(r, kF_Csel_S_S, ValuesS(), false);
    ScalarConditional(r, kF_Csel_S_D, ValuesD(), false);
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: saturating shift right narrow", "[a64][unicorn][ryujinx]") {
    Runner r;
    // Scalar (Rd = 0, Rn = 1) and vector (Q both ways) over Ryujinx's integer edge values.
    const auto narrow = [&](const auto& opcodes, const auto& values, u32 esize, bool vector) {
        for (u32 opcode : opcodes) {
            for (const u32 shift : {1u, esize / 2}) {
                for (const u32 q : {0u, 1u}) {
                    if (q && !vector)
                        continue;
                    const u32 inst = opcode | (((esize - shift) & 0x7F) << 16) | (1 << 5) | (q << 30);
                    for (const u64 a : values) {
                        for (const u64 b : values) {
                            r.Run(inst, {0}, [&](A64FuzzCase& c) {
                                c.vecs[0] = {Garbage(), Garbage()};
                                c.vecs[1] = {a, vector ? b : 0};
                            });
                        }
                    }
                }
            }
        }
    };
    narrow(kShrImmSaturatingNarrow_S_HB, k1H, 16, false);
    narrow(kShrImmSaturatingNarrow_S_SH, k1S, 32, false);
    narrow(kShrImmSaturatingNarrow_S_DS, k1D, 64, false);
    narrow(kShrImmSaturatingNarrow_V_8H8B_8H16B, k4H, 16, true);
    narrow(kShrImmSaturatingNarrow_V_4S4H_4S8H, k2S, 32, true);
    narrow(kShrImmSaturatingNarrow_V_2D2S_2D4S, k1D, 64, true);
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: SDIV UDIV", "[a64][unicorn][ryujinx]") {
    Runner r;
    // CpuTestAluBinary.cs: Rd in {0, 31}, Rn in {1, 31}, Rm in {2, 31}, over the edge values.
    constexpr std::array<u64, 4> w_values{0x00000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF};
    constexpr std::array<u64, 4> x_values{0x0000000000000000, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF};
    const auto run = [&](u32 opcode, const auto& values) {
        for (const u32 rd : {0u, 31u}) {
            for (const u32 rn : {1u, 31u}) {
                for (const u32 rm : {2u, 31u}) {
                    for (const u64 n : values) {
                        for (const u64 m : values) {
                            r.Run(opcode | (rm << 16) | (rn << 5) | rd, {0}, [&](A64FuzzCase& c) {
                                c.regs[1] = n;
                                c.regs[2] = m;
                            });
                        }
                    }
                }
            }
        }
    };
    run(0x9AC00C00, x_values);  // SDIV X0, X0, X0
    run(0x1AC00C00, w_values);  // SDIV W0, W0, W0
    run(0x9AC00800, x_values);  // UDIV X0, X0, X0
    run(0x1AC00800, w_values);  // UDIV W0, W0, W0
    REQUIRE(r.mismatches == 0);
}

TEST_CASE("Ryujinx A64: CRC32 CRC32C", "[a64][unicorn][ryujinx]") {
    Runner r;
    // CpuTestAluBinary.cs, against Unicorn: B/H/W/X, both polynomials, Wn in {0, ~0}.
    constexpr std::array<u64, 4> xm{0x0000000000000000, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF};
    for (const u32 c : {0u, 1u}) {
        for (u32 size = 0; size < 4; ++size) {
            const u32 opcode = (size == 3 ? 0x9AC04000 : 0x1AC04000) | (c << 12) | (size << 10);
            for (const u32 rd : {0u, 31u}) {
                for (const u32 rn : {1u, 31u}) {
                    for (const u32 rm : {2u, 31u}) {
                        for (const u64 wn : {u64(0), u64(0xFFFFFFFF)}) {
                            for (const u64 m : xm) {
                                const u64 width_mask = size == 3 ? ~u64(0) : (u64(1) << (8 << size)) - 1;
                                r.Run(opcode | (rm << 16) | (rn << 5) | rd, {0}, [&](A64FuzzCase& state) {
                                    state.regs[1] = wn;
                                    state.regs[2] = (m & width_mask) | (size == 3 ? 0 : Garbage() & ~width_mask);
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    REQUIRE(r.mismatches == 0);

    // Ryujinx's known answers for the CRC32 (not CRC32C) forms, sizes B, H, W, X.
    struct CrcTest {
        u32 crc;
        u64 value;
        std::array<u32, 4> results;
    };
    constexpr std::array<CrcTest, 10> known{{
        {0x00000000, 0x0000000000000000, {0x00000000, 0x00000000, 0x00000000, 0x00000000}},
        {0x00000000, 0x7fffffffffffffff, {0x2d02ef8d, 0xbe2612ff, 0xdebb20e3, 0xa9de8355}},
        {0x00000000, 0x8000000000000000, {0x00000000, 0x00000000, 0x00000000, 0xedb88320}},
        {0x00000000, 0xffffffffffffffff, {0x2d02ef8d, 0xbe2612ff, 0xdebb20e3, 0x44660075}},
        {0x00000000, 0xa002f1ca52788c1c, {0x14015c4f, 0x02799256, 0x9063c9e5, 0x8816610a}},
        {0xffffffff, 0x0000000000000000, {0x2dfd1072, 0xbe26ed00, 0xdebb20e3, 0x9add2096}},
        {0xffffffff, 0x7fffffffffffffff, {0x00ffffff, 0x0000ffff, 0x00000000, 0x3303a3c3}},
        {0xffffffff, 0x8000000000000000, {0x2dfd1072, 0xbe26ed00, 0xdebb20e3, 0x7765a3b6}},
        {0xffffffff, 0xffffffffffffffff, {0x00ffffff, 0x0000ffff, 0x00000000, 0xdebb20e3}},
        {0xffffffff, 0xa002f1ca52788c1c, {0x39fc4c3d, 0xbc5f7f56, 0x4ed8e906, 0x12cb419c}},
    }};
    A64TestEnv env;
    Dynarmic::A64::UserConfig conf{GetUserConfig(env)};
    Dynarmic::A64::Jit jit{conf};
    for (const auto& test : known) {
        for (u32 size = 0; size < 4; ++size) {
            env.code_mem = {(size == 3 ? 0x9AC24020u : 0x1AC24020u) | (size << 10), 0x14000000};  // CRC32<sz> W0, W1, <R>2; B .
            env.code_mem_start_address = 0;
            jit.SetRegister(1, test.crc);
            jit.SetRegister(2, test.value);
            jit.SetPC(0);
            jit.ClearCache();
            env.ticks_left = 2;
            jit.Run();
            INFO("crc " << std::hex << test.crc << " value " << test.value << " size " << size);
            CHECK(jit.GetRegister(0) == test.results[size]);
        }
    }
}
