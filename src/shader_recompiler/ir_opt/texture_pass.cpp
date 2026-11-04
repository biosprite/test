// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <optional>
#include "common/container/unordered_map.h"
#include <tuple>
#include <limits>
#include <boost/container/small_vector.hpp>

#include "common/settings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Optimization {
namespace {
struct TextureInst {
    ConstBufferAddr cbuf;
    IR::Inst* inst;
    IR::Block* block;
};

using TextureInstVector = boost::container::small_vector<TextureInst, 24>;

const IR::Block* ConditionalHeader(const IR::Block* body) {
    const auto predecessors{body->ImmPredecessors()};
    const auto successors{body->ImmSuccessors()};
    if (predecessors.size() != 1 || successors.size() != 1) return nullptr;
    const IR::Block* const header{predecessors[0]};
    const auto targets{header->ImmSuccessors()};
    if (targets.size() != 2 || targets[0] != body || targets[1] != successors[0] ||
        targets[1]->ImmPredecessors().size() != 2 || header->empty() ||
        header->back().GetOpcode() != IR::Opcode::ConditionRef) {
        return nullptr;
    }
    return header;
}

std::pair<IR::Value, bool> Condition(IR::Value value) {
    bool negated{};
    while (const IR::Inst* inst = value.TryInstRecursive()) {
        if (inst->GetOpcode() == IR::Opcode::LogicalNot) negated = !negated;
        else if (inst->GetOpcode() != IR::Opcode::ConditionRef) break;
        value = inst->Arg(0);
    }
    return {value.Resolve(), negated};
}

IR::Value ResolveConditionalPhi(IR::Value value, const IR::Block* block) {
    if (!value.Resolve().IsPhi() || !(block = ConditionalHeader(block))) return value;
    const auto [condition, negated]{Condition(block->back().Arg(0))};
    boost::container::small_vector<const IR::Block*, 16> visited;
    while (value.Resolve().IsPhi() && std::ranges::find(visited, block) == visited.end()) {
        visited.push_back(block);
        const IR::Inst* const phi{value.InstRecursive()};
        const auto predecessors{block->ImmPredecessors()};
        if (predecessors.size() == 1) {
            block = predecessors[0];
            continue;
        }
        if (predecessors.size() != 2 || phi->NumArgs() != 2) break;
        const IR::Block* body{predecessors[0]};
        const IR::Block* header{ConditionalHeader(body)};
        if (header != predecessors[1]) {
            body = predecessors[1];
            header = ConditionalHeader(body);
            if (header != predecessors[0]) break;
        }
        const auto [other_condition, other_negated]{Condition(header->back().Arg(0))};
        if (condition == other_condition && (phi->PhiBlock(0) == header || phi->PhiBlock(0) == body)) {
            const IR::Block* const incoming{negated == other_negated ? body : header};
            value = phi->Arg(phi->PhiBlock(0) == incoming ? 0 : 1);
        }
        block = header;
    }
    return value;
}

constexpr u32 DESCRIPTOR_SIZE = 8;
constexpr u32 DESCRIPTOR_SIZE_SHIFT = u32(std::countr_zero(DESCRIPTOR_SIZE));
constexpr u32 DESCRIPTOR_MAX_COUNT = 1024;
/// Upper bound of elements read from a table of handles that lives in a storage buffer, the
/// host has no size information for the array so the shader index is clamped to this
constexpr u32 STORAGE_DESCRIPTOR_MAX_COUNT = 32;
/// Marks a handle that comes from global memory with an addressing the host can't resolve. It
/// stops the search so it doesn't fall through to the storage buffer pointer words.
constexpr u32 UNTRACKABLE_STORAGE_HANDLE = (std::numeric_limits<u32>::max)();

ConstBufferAddr MakeUntrackable() {
    return ConstBufferAddr{
        .index = UNTRACKABLE_STORAGE_HANDLE,
        .offset = 0,
        .shift_left = 0,
        .secondary_index = 0,
        .secondary_offset = 0,
        .secondary_shift_left = 0,
        .dynamic_offset = {},
        .count = 1,
        .has_secondary = false,
        .storage_offset = 0,
        .storage_stride = 0,
        .is_storage = true,
    };
}

bool IsUntrackable(const ConstBufferAddr& addr) {
    return addr.index == UNTRACKABLE_STORAGE_HANDLE;
}

u32 DynamicDescriptorSizeShift(const IR::U32& dynamic_offset) {
    const IR::Inst* const inst = dynamic_offset.InstRecursive();
    if (!inst || inst->GetOpcode() != IR::Opcode::ShiftLeftLogical32)
        return DESCRIPTOR_SIZE_SHIFT;
    const IR::Value shift = inst->Arg(1);
    if (!shift.IsImmediate())
        return DESCRIPTOR_SIZE_SHIFT;
    const u32 size_shift = shift.U32();
    return size_shift >= DESCRIPTOR_SIZE_SHIFT && size_shift < 31 ? size_shift : DESCRIPTOR_SIZE_SHIFT;
}

struct StorageTableIndex {
    IR::U32 index;  ///< Element index expression
    u32 stride;     ///< Element stride in bytes
};

/// Splits the dynamic byte offset of a table of handles in a storage buffer into its index and
/// stride: idx << shift or idx * stride. Unlike constant buffer tables the stride is arbitrary.
std::optional<StorageTableIndex> SplitStorageTableIndex(const IR::U32& dynamic_offset) {
    const IR::Inst* const inst = dynamic_offset.TryInstRecursive();
    if (!inst) {
        return std::nullopt;
    }
    const IR::Value stride{inst->Arg(1)};
    if (!stride.IsImmediate()) {
        return std::nullopt;
    }
    u32 stride_bytes{};
    if (inst->GetOpcode() == IR::Opcode::ShiftLeftLogical32) {
        if (stride.U32() >= 31) {
            return std::nullopt;
        }
        stride_bytes = 1U << stride.U32();
    } else if (inst->GetOpcode() == IR::Opcode::IMul32) {
        stride_bytes = stride.U32();
    } else {
        return std::nullopt;
    }
    if (stride_bytes < sizeof(u32) || stride_bytes % sizeof(u32) != 0) {
        return std::nullopt;
    }
    return StorageTableIndex{
        .index = IR::U32{inst->Arg(0)},
        .stride = stride_bytes,
    };
}

u32 DynamicDescriptorCount(u32 base_offset, u32 size_shift, u32 max_descriptors) {
    auto const descriptor_limit = (std::max)(1U, max_descriptors);
    auto const max_cbuf_bytes = 16 * descriptor_limit;
    if (size_shift >= 31 || base_offset >= max_cbuf_bytes)
        return 1;
    auto const stride = 1U << size_shift;
    auto const available = max_cbuf_bytes - base_offset;
    if (available < DESCRIPTOR_SIZE)
        return 1;
    auto const available_count = 1U + (available - DESCRIPTOR_SIZE) / stride;
    return std::min(descriptor_limit, available_count);
}

u32 SaturatingSub(u32 lhs, u32 rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

template <typename T>
[[nodiscard]] u32 StaticDescriptorCount(T const& descriptors) noexcept {
    return std::accumulate(descriptors.cbegin(), descriptors.cend(), 0U, [](auto const& acc, auto const& e) {
        return acc + (e.count <= 1 ? e.count : 0);
    });
}

u32 DynamicSampledTextureCap(const Info& info, const HostTranslateInfo& host_info, u32 dynamic_arrays) {
    auto const sampled_limit = (std::max)(1U, std::min(host_info.max_per_stage_descriptor_sampled_images,
                                                       host_info.max_descriptor_set_sampled_images));
    auto const resource_limit = (std::max)(1U, host_info.max_per_stage_resources);
    if (dynamic_arrays > 0) {
        auto const sampled_static_count = StaticDescriptorCount(info.texture_buffer_descriptors) + StaticDescriptorCount(info.texture_descriptors);
        auto const resource_static_count =
            NumDescriptors(info.constant_buffer_descriptors)
            + NumDescriptors(info.storage_buffers_descriptors)
            + sampled_static_count + NumDescriptors(info.image_buffer_descriptors)
            + NumDescriptors(info.image_descriptors);
        auto const sampled_budget = SaturatingSub(sampled_limit, sampled_static_count);
        auto const resource_budget = SaturatingSub(resource_limit, resource_static_count);
        auto const sampled_cap = sampled_budget / dynamic_arrays;
        auto const resource_cap = resource_budget / dynamic_arrays;
        return (std::max)(1U, (std::min)(sampled_cap, resource_cap));
    }
    return (std::min)({DESCRIPTOR_MAX_COUNT, sampled_limit, resource_limit});
}

IR::Opcode IndexedInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::BindlessImageSampleImplicitLod:
    case IR::Opcode::BoundImageSampleImplicitLod:
        return IR::Opcode::ImageSampleImplicitLod;
    case IR::Opcode::BoundImageSampleExplicitLod:
    case IR::Opcode::BindlessImageSampleExplicitLod:
        return IR::Opcode::ImageSampleExplicitLod;
    case IR::Opcode::BoundImageSampleDrefImplicitLod:
    case IR::Opcode::BindlessImageSampleDrefImplicitLod:
        return IR::Opcode::ImageSampleDrefImplicitLod;
    case IR::Opcode::BoundImageSampleDrefExplicitLod:
    case IR::Opcode::BindlessImageSampleDrefExplicitLod:
        return IR::Opcode::ImageSampleDrefExplicitLod;
    case IR::Opcode::BindlessImageGather:
    case IR::Opcode::BoundImageGather:
        return IR::Opcode::ImageGather;
    case IR::Opcode::BindlessImageGatherDref:
    case IR::Opcode::BoundImageGatherDref:
        return IR::Opcode::ImageGatherDref;
    case IR::Opcode::BindlessImageFetch:
    case IR::Opcode::BoundImageFetch:
        return IR::Opcode::ImageFetch;
    case IR::Opcode::BoundImageQueryDimensions:
    case IR::Opcode::BindlessImageQueryDimensions:
        return IR::Opcode::ImageQueryDimensions;
    case IR::Opcode::BoundImageQueryLod:
    case IR::Opcode::BindlessImageQueryLod:
        return IR::Opcode::ImageQueryLod;
    case IR::Opcode::BoundImageGradient:
    case IR::Opcode::BindlessImageGradient:
        return IR::Opcode::ImageGradient;
    case IR::Opcode::BoundImageRead:
    case IR::Opcode::BindlessImageRead:
        return IR::Opcode::ImageRead;
    case IR::Opcode::BoundImageWrite:
    case IR::Opcode::BindlessImageWrite:
        return IR::Opcode::ImageWrite;
    case IR::Opcode::BoundImageAtomicIAdd32:
    case IR::Opcode::BindlessImageAtomicIAdd32:
        return IR::Opcode::ImageAtomicIAdd32;
    case IR::Opcode::BoundImageAtomicSMin32:
    case IR::Opcode::BindlessImageAtomicSMin32:
        return IR::Opcode::ImageAtomicSMin32;
    case IR::Opcode::BoundImageAtomicUMin32:
    case IR::Opcode::BindlessImageAtomicUMin32:
        return IR::Opcode::ImageAtomicUMin32;
    case IR::Opcode::BoundImageAtomicSMax32:
    case IR::Opcode::BindlessImageAtomicSMax32:
        return IR::Opcode::ImageAtomicSMax32;
    case IR::Opcode::BoundImageAtomicUMax32:
    case IR::Opcode::BindlessImageAtomicUMax32:
        return IR::Opcode::ImageAtomicUMax32;
    case IR::Opcode::BoundImageAtomicInc32:
    case IR::Opcode::BindlessImageAtomicInc32:
        return IR::Opcode::ImageAtomicInc32;
    case IR::Opcode::BoundImageAtomicDec32:
    case IR::Opcode::BindlessImageAtomicDec32:
        return IR::Opcode::ImageAtomicDec32;
    case IR::Opcode::BoundImageAtomicAnd32:
    case IR::Opcode::BindlessImageAtomicAnd32:
        return IR::Opcode::ImageAtomicAnd32;
    case IR::Opcode::BoundImageAtomicOr32:
    case IR::Opcode::BindlessImageAtomicOr32:
        return IR::Opcode::ImageAtomicOr32;
    case IR::Opcode::BoundImageAtomicXor32:
    case IR::Opcode::BindlessImageAtomicXor32:
        return IR::Opcode::ImageAtomicXor32;
    case IR::Opcode::BoundImageAtomicExchange32:
    case IR::Opcode::BindlessImageAtomicExchange32:
        return IR::Opcode::ImageAtomicExchange32;
    default:
        return IR::Opcode::Void;
    }
}

bool IsStorageImageOpcode(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::ImageRead:
    case IR::Opcode::ImageAtomicIAdd32:
    case IR::Opcode::ImageAtomicSMin32:
    case IR::Opcode::ImageAtomicUMin32:
    case IR::Opcode::ImageAtomicSMax32:
    case IR::Opcode::ImageAtomicUMax32:
    case IR::Opcode::ImageAtomicInc32:
    case IR::Opcode::ImageAtomicDec32:
    case IR::Opcode::ImageAtomicAnd32:
    case IR::Opcode::ImageAtomicOr32:
    case IR::Opcode::ImageAtomicXor32:
    case IR::Opcode::ImageAtomicExchange32:
    case IR::Opcode::ImageWrite:
        return true;
    default:
        return false;
    }
}

u32 DynamicSampledTextureArrayCount(const TextureInstVector& to_replace) {
    u32 count{};
    for (const TextureInst& inst : to_replace) {
        const auto flags{inst.inst->Flags<IR::TextureInstInfo>()};
        if (inst.cbuf.count > 1 && !IsStorageImageOpcode(IndexedInstruction(*inst.inst)) &&
            flags.type != TextureType::Buffer) {
            ++count;
        }
    }
    return count;
}

bool IsBindless(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::BindlessImageSampleImplicitLod:
    case IR::Opcode::BindlessImageSampleExplicitLod:
    case IR::Opcode::BindlessImageSampleDrefImplicitLod:
    case IR::Opcode::BindlessImageSampleDrefExplicitLod:
    case IR::Opcode::BindlessImageGather:
    case IR::Opcode::BindlessImageGatherDref:
    case IR::Opcode::BindlessImageFetch:
    case IR::Opcode::BindlessImageQueryDimensions:
    case IR::Opcode::BindlessImageQueryLod:
    case IR::Opcode::BindlessImageGradient:
    case IR::Opcode::BindlessImageRead:
    case IR::Opcode::BindlessImageWrite:
    case IR::Opcode::BindlessImageAtomicIAdd32:
    case IR::Opcode::BindlessImageAtomicSMin32:
    case IR::Opcode::BindlessImageAtomicUMin32:
    case IR::Opcode::BindlessImageAtomicSMax32:
    case IR::Opcode::BindlessImageAtomicUMax32:
    case IR::Opcode::BindlessImageAtomicInc32:
    case IR::Opcode::BindlessImageAtomicDec32:
    case IR::Opcode::BindlessImageAtomicAnd32:
    case IR::Opcode::BindlessImageAtomicOr32:
    case IR::Opcode::BindlessImageAtomicXor32:
    case IR::Opcode::BindlessImageAtomicExchange32:
        return true;
    case IR::Opcode::BoundImageSampleImplicitLod:
    case IR::Opcode::BoundImageSampleExplicitLod:
    case IR::Opcode::BoundImageSampleDrefImplicitLod:
    case IR::Opcode::BoundImageSampleDrefExplicitLod:
    case IR::Opcode::BoundImageGather:
    case IR::Opcode::BoundImageGatherDref:
    case IR::Opcode::BoundImageFetch:
    case IR::Opcode::BoundImageQueryDimensions:
    case IR::Opcode::BoundImageQueryLod:
    case IR::Opcode::BoundImageGradient:
    case IR::Opcode::BoundImageRead:
    case IR::Opcode::BoundImageWrite:
    case IR::Opcode::BoundImageAtomicIAdd32:
    case IR::Opcode::BoundImageAtomicSMin32:
    case IR::Opcode::BoundImageAtomicUMin32:
    case IR::Opcode::BoundImageAtomicSMax32:
    case IR::Opcode::BoundImageAtomicUMax32:
    case IR::Opcode::BoundImageAtomicInc32:
    case IR::Opcode::BoundImageAtomicDec32:
    case IR::Opcode::BoundImageAtomicAnd32:
    case IR::Opcode::BoundImageAtomicOr32:
    case IR::Opcode::BoundImageAtomicXor32:
    case IR::Opcode::BoundImageAtomicExchange32:
        return false;
    default:
        throw InvalidArgument("Invalid opcode {}", inst.GetOpcode());
    }
}

bool IsTextureInstruction(const IR::Inst& inst) {
    return IndexedInstruction(inst) != IR::Opcode::Void;
}
// Per-pass caches

static inline u32 ReadCbufCached(Environment& env, u32 index, u32 offset) {
    const CbufWordKey k{index, offset};
    if (auto it = env.cbuf_word_cache.find(k); it != env.cbuf_word_cache.end())
        return it->second;
    const u32 v = env.ReadCbufValue(index, offset);
    env.cbuf_word_cache.emplace(k, v);
    return v;
}

static inline bool IsAttributeHandle(const ConstBufferAddr& cbuf) {
    return cbuf.index == ATTRIBUTE_HANDLE_CBUF_INDEX;
}

/// Reads the first element of a handle stored in global memory: the storage buffer address is
/// read from its {u64 address, u32 size} descriptor at (cbuf_index, cbuf_offset)
static inline u32 ReadStorageHandleCached(Environment& env, u32 cbuf_index, u32 cbuf_offset,
                                          s32 storage_offset) {
    const u64 address_low{ReadCbufCached(env, cbuf_index, cbuf_offset)};
    const u64 address_high{ReadCbufCached(env, cbuf_index, cbuf_offset + 4)};
    const u64 storage_addr{address_low | (address_high << 32)};
    if (storage_addr == 0) {
        return 0;
    }
    return env.ReadGlobalU32(storage_addr + static_cast<u64>(s64{storage_offset}));
}

/// Reads the word the shader ORs into the handle: an immediate, a constant buffer word or a word
/// stored in a storage buffer
static inline u32 ReadSecondaryHandleCached(Environment& env, const ConstBufferAddr& cbuf) {
    if (cbuf.secondary_index == IMMEDIATE_HANDLE_CBUF_INDEX) {
        return cbuf.secondary_offset;
    }
    if (cbuf.secondary_is_storage) {
        return ReadStorageHandleCached(env, cbuf.secondary_index, cbuf.secondary_offset,
                                       cbuf.secondary_storage_offset);
    }
    return ReadCbufCached(env, cbuf.secondary_index, cbuf.secondary_offset);
}

static inline u32 GetTextureHandleCached(Environment& env, const ConstBufferAddr& cbuf) {
    const HandleKey hk{cbuf.index,          cbuf.offset,        cbuf.shift_left,
                       cbuf.secondary_index, cbuf.secondary_offset, cbuf.secondary_shift_left,
                       cbuf.count,          cbuf.has_secondary, cbuf.storage_offset,
                       cbuf.is_storage,     cbuf.secondary_storage_offset,
                       cbuf.secondary_is_storage};
    if (auto it = env.handle_cache.find(hk); it != env.handle_cache.end())
        return it->second;
    u32 handle{};
    if (cbuf.is_storage) {
        handle = ReadStorageHandleCached(env, cbuf.index, cbuf.offset, cbuf.storage_offset)
                 << cbuf.shift_left;
    } else {
        handle = ReadCbufCached(env, cbuf.index, cbuf.offset) << cbuf.shift_left;
    }
    if (cbuf.has_secondary) {
        handle |= ReadSecondaryHandleCached(env, cbuf) << cbuf.secondary_shift_left;
    } else if (!cbuf.is_storage) {
        // Without a secondary word the unshifted word is ORed back in
        handle |= ReadCbufCached(env, cbuf.index, cbuf.offset);
    }
    env.handle_cache.emplace(hk, handle);
    return handle;
}

// Cached variants of existing helpers
static inline TextureType ReadTextureTypeCached(Environment& env, const ConstBufferAddr& cbuf) {
    if (IsAttributeHandle(cbuf)) {
        return TextureType::Color2D;
    }
    return env.ReadTextureType(GetTextureHandleCached(env, cbuf));
}
static inline TexturePixelFormat ReadTexturePixelFormatCached(Environment& env,
                                                                const ConstBufferAddr& cbuf) {
    if (IsAttributeHandle(cbuf)) {
        return TexturePixelFormat::A8B8G8R8_UNORM;
    }
    return env.ReadTexturePixelFormat(GetTextureHandleCached(env, cbuf));
}
static inline bool IsTexturePixelFormatIntegerCached(Environment& env,
                                                        const ConstBufferAddr& cbuf) {
    if (IsAttributeHandle(cbuf)) {
        return false;
    }
    return env.IsTexturePixelFormatInteger(GetTextureHandleCached(env, cbuf));
}


std::optional<ConstBufferAddr> Track(const IR::Value& value, Environment& env, const HostTranslateInfo& host_info, const IR::Block* block, const Info& info);
static inline std::optional<ConstBufferAddr> TrackCached(const IR::Value& v, Environment& env, const HostTranslateInfo& host_info, const IR::Block* block, const Info& info) {
    if (const IR::Inst* key = v.TryInstRecursive()) {
        if (auto it = env.track_cache.find(key); it != env.track_cache.end()) return it->second;
        auto found = Track(v, env, host_info, block, info);
        if (found) env.track_cache.emplace(key, *found);
        return found;
    }
    return Track(v, env, host_info, block, info);
}

std::optional<ConstBufferAddr> TryGetConstBuffer(const IR::Inst* inst, Environment& env, const HostTranslateInfo& host_info, const IR::Block* block, const Info& info);

std::optional<ConstBufferAddr> Track(const IR::Value& value, Environment& env, const HostTranslateInfo& host_info, const IR::Block* block, const Info& info) {
    return IR::BreadthFirstSearch(value, [&env, &host_info, block, &info](const IR::Inst* inst) {
        return TryGetConstBuffer(inst, env, host_info, block, info);
    }, [&](IR::Value operand) { return ResolveConditionalPhi(operand, block); });
}

/// Returns true when value is the low 32 bits of the storage buffer address read from the
/// constant buffer word that the storage buffer pass tracked
bool IsStorageAddressLow(const IR::Value& value, const StorageBufferDescriptor& ssbo) {
    const IR::Inst* inst{value.TryInstRecursive()};
    if (!inst) {
        return false;
    }
    if (inst->GetOpcode() == IR::Opcode::CompositeExtractU32x2) {
        const IR::Value element{inst->Arg(1)};
        if (!element.IsImmediate() || element.U32() != 0) {
            return false;
        }
        inst = inst->Arg(0).TryInstRecursive();
        if (!inst || inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return false;
        }
    } else if (inst->GetOpcode() != IR::Opcode::GetCbufU32) {
        return false;
    }
    const IR::Value index{inst->Arg(0)};
    const IR::Value offset{inst->Arg(1)};
    return index.IsImmediate() && offset.IsImmediate() && index.U32() == ssbo.cbuf_index &&
           offset.U32() == ssbo.cbuf_offset;
}

/// Matches the aligned storage buffer base the storage buffer pass subtracts from addresses:
/// (address_low & ~(alignment - 1)) or plain address_low when no alignment is needed
bool IsAlignedStorageAddress(const IR::Value& value, const StorageBufferDescriptor& ssbo) {
    if (IsStorageAddressLow(value, ssbo)) {
        return true;
    }
    const IR::Inst* const inst{value.TryInstRecursive()};
    if (!inst || inst->GetOpcode() != IR::Opcode::BitwiseAnd32) {
        return false;
    }
    return inst->Arg(1).IsImmediate() && IsStorageAddressLow(inst->Arg(0), ssbo);
}

/// Compares two resolved values, IR::Value::operator== is not usable on vector immediates
bool SameValue(const IR::Value& lhs, const IR::Value& rhs) {
    if (lhs.IsImmediate() != rhs.IsImmediate()) {
        return false;
    }
    if (!lhs.IsImmediate()) {
        return lhs.InstRecursive() == rhs.InstRecursive();
    }
    if (lhs.Type() != rhs.Type()) {
        return false;
    }
    switch (lhs.Type()) {
    case IR::Type::U32:
        return lhs.U32() == rhs.U32();
    case IR::Type::U64:
        return lhs.U64() == rhs.U64();
    default:
        return false;
    }
}

using PhiStack = boost::container::small_vector<const IR::Inst*, 8>;

std::optional<IR::Value> ResolveInvariantPhiImpl(const IR::Inst* phi, PhiStack& visited) {
    if (visited.size() >= 8) {
        return std::nullopt;
    }
    visited.push_back(phi);
    std::optional<IR::Value> incoming;
    for (size_t arg = 0; arg < phi->NumArgs(); ++arg) {
        IR::Value value{phi->Arg(arg).Resolve()};
        if (const IR::Inst* const inst{value.TryInstRecursive()};
            inst && inst->GetOpcode() == IR::Opcode::Phi) {
            if (std::ranges::find(visited, inst) != visited.end()) {
                // Back edge, it carries the value of the phi being resolved
                continue;
            }
            const std::optional<IR::Value> resolved{ResolveInvariantPhiImpl(inst, visited)};
            if (!resolved) {
                return std::nullopt;
            }
            value = *resolved;
        }
        if (incoming && !SameValue(*incoming, value)) {
            return std::nullopt;
        }
        incoming = value;
    }
    return incoming;
}

/// Looks through phi nodes whose incoming values are all the same value defined outside of the
/// loop (the back edges only carry the phi itself), the storage buffer address of a table of
/// handles read inside a loop reaches the load this way
IR::Value ResolveInvariantPhi(const IR::Value& value) {
    const IR::Inst* const inst{value.TryInstRecursive()};
    if (!inst || inst->GetOpcode() != IR::Opcode::Phi) {
        return value;
    }
    PhiStack visited;
    return ResolveInvariantPhiImpl(inst, visited).value_or(value);
}

struct StorageAddressTerms {
    bool has_base{};      ///< The storage buffer address was found in the expression
    u32 immediate{};      ///< Sum of the immediate byte offsets
    IR::U32 dynamic;      ///< Optional dynamic byte offset (idx << shift or idx * stride)
};

/// Decomposes the address passed to a global memory load into base + immediates + dynamic index
bool CollectStorageAddressTerms(const IR::Value& raw_value, const StorageBufferDescriptor& ssbo,
                                StorageAddressTerms& terms, u32 depth) {
    if (depth > 16) {
        return false;
    }
    const IR::Value value{ResolveInvariantPhi(raw_value)};
    if (value.IsImmediate()) {
        terms.immediate +=
            value.Type() == IR::Type::U64 ? static_cast<u32>(value.U64()) : value.U32();
        return true;
    }
    if (IsStorageAddressLow(value, ssbo)) {
        if (terms.has_base) {
            return false;
        }
        terms.has_base = true;
        return true;
    }
    const IR::Inst* const inst{value.InstRecursive()};
    switch (inst->GetOpcode()) {
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
        return CollectStorageAddressTerms(inst->Arg(0), ssbo, terms, depth + 1) &&
               CollectStorageAddressTerms(inst->Arg(1), ssbo, terms, depth + 1);
    case IR::Opcode::ConvertU32U64:
    case IR::Opcode::ConvertU64U32:
        return CollectStorageAddressTerms(inst->Arg(0), ssbo, terms, depth + 1);
    case IR::Opcode::PackUint2x32: {
        const IR::Inst* const vector{inst->Arg(0).TryInstRecursive()};
        if (!vector || vector->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
            return false;
        }
        // Only the low word matters, the high word carries the address high bits (and carry)
        return CollectStorageAddressTerms(vector->Arg(0), ssbo, terms, depth + 1);
    }
    default:
        if (value.Type() != IR::Type::U32 || !terms.dynamic.IsEmpty()) {
            return false;
        }
        terms.dynamic = IR::U32{value};
        return true;
    }
}

/// Tracks a bindless handle loaded from a storage buffer back to the constant buffer words that
/// hold the storage buffer address, so the host can read the handle itself when binding
std::optional<ConstBufferAddr> TrackStorageHandle(const IR::Inst& load, u32 element_offset,
                                                  const Info& info) {
    const IR::Value binding{load.Arg(0)};
    const IR::Value offset{load.Arg(1)};
    if (!binding.IsImmediate() || binding.U32() >= info.storage_buffers_descriptors.size()) {
        return MakeUntrackable();
    }
    const StorageBufferDescriptor& ssbo{info.storage_buffers_descriptors[binding.U32()]};
    StorageAddressTerms terms;
    terms.immediate = element_offset;
    if (offset.IsImmediate()) {
        terms.has_base = true;
        terms.immediate += offset.U32();
    } else {
        // The storage buffer pass emits: (address_low [+ offsets]) - (address_low & ~align_mask)
        const IR::Inst* const sub{offset.InstRecursive()};
        if (sub->GetOpcode() != IR::Opcode::ISub32 || !IsAlignedStorageAddress(sub->Arg(1), ssbo) ||
            !CollectStorageAddressTerms(sub->Arg(0), ssbo, terms, 0) || !terms.has_base) {
            return MakeUntrackable();
        }
    }
    u32 count{1};
    u32 stride{0};
    IR::U32 table_index;
    if (!terms.dynamic.IsEmpty()) {
        const std::optional<StorageTableIndex> table{SplitStorageTableIndex(terms.dynamic)};
        if (!table) {
            return MakeUntrackable();
        }
        // The shader indexes the descriptor array directly with the element index, so unlike
        // constant buffer tables no size shift is applied to dynamic_offset
        table_index = table->index;
        stride = table->stride;
        count = STORAGE_DESCRIPTOR_MAX_COUNT;
    }
    return ConstBufferAddr{
        .index = ssbo.cbuf_index,
        .offset = ssbo.cbuf_offset,
        .shift_left = 0,
        .secondary_index = 0,
        .secondary_offset = 0,
        .secondary_shift_left = 0,
        .dynamic_offset = table_index,
        .count = count,
        .has_secondary = false,
        .storage_offset = static_cast<s32>(terms.immediate),
        .storage_stride = stride,
        .is_storage = true,
    };
}

std::optional<u32> TryGetConstant(IR::Value& value, Environment& env) {
    const IR::Inst* inst = value.InstRecursive();
    if (inst->GetOpcode() != IR::Opcode::GetCbufU32) {
        return std::nullopt;
    }
    const IR::Value index{inst->Arg(0)};
    const IR::Value offset{inst->Arg(1)};
    if (!index.IsImmediate()) {
        return std::nullopt;
    }
    if (!offset.IsImmediate()) {
        return std::nullopt;
    }
    const auto index_number = index.U32();
    if (index_number != 1) {
        return std::nullopt;
    }
    const auto offset_number = offset.U32();
    return ReadCbufCached(env, index_number, offset_number);
}

std::optional<ConstBufferAddr> TryGetConstBuffer(const IR::Inst* inst, Environment& env, const HostTranslateInfo& host_info, const IR::Block* block, const Info& info) {
    switch (inst->GetOpcode()) {
    default:
        return std::nullopt;
    case IR::Opcode::LoadStorage32:
        return TrackStorageHandle(*inst, 0, info);
    case IR::Opcode::LoadStorageU8:
    case IR::Opcode::LoadStorageS8:
    case IR::Opcode::LoadStorageU16:
    case IR::Opcode::LoadStorageS16:
    case IR::Opcode::LoadStorage64:
    case IR::Opcode::LoadStorage128:
        // Handles are 32-bit words, 64 and 128 bit loads are tracked through CompositeExtract
        return MakeUntrackable();
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x4: {
        const IR::Inst* const vector{inst->Arg(0).TryInstRecursive()};
        if (!vector || (vector->GetOpcode() != IR::Opcode::LoadStorage64 &&
                        vector->GetOpcode() != IR::Opcode::LoadStorage128)) {
            return std::nullopt;
        }
        const IR::Value element{inst->Arg(1)};
        if (!element.IsImmediate()) {
            return MakeUntrackable();
        }
        return TrackStorageHandle(*vector, element.U32() * sizeof(u32), info);
    }
    case IR::Opcode::BitwiseOr32: {
        const IR::Value op1{inst->Arg(0)};
        const IR::Value op2{inst->Arg(1)};
        if (op1.IsImmediate() != op2.IsImmediate()) {
            // word | immediate, the immediate usually carries the sampler index
            const u32 immediate{(op1.IsImmediate() ? op1 : op2).U32()};
            std::optional lhs{
                TrackCached(op1.IsImmediate() ? op2 : op1, env, host_info, block, info)};
            if (!lhs || IsUntrackable(*lhs)) {
                return lhs;
            }
            if (lhs->has_secondary) {
                return lhs->is_storage ? std::optional{MakeUntrackable()} : std::nullopt;
            }
            // Tables (count > 1) are fine, the immediate is the same for every element
            lhs->secondary_index = IMMEDIATE_HANDLE_CBUF_INDEX;
            lhs->secondary_offset = immediate;
            lhs->secondary_shift_left = 0;
            lhs->secondary_storage_offset = 0;
            lhs->secondary_is_storage = false;
            lhs->has_secondary = true;
            return lhs;
        }
        std::optional lhs{TrackCached(op1, env, host_info, block, info)};
        std::optional rhs{TrackCached(op2, env, host_info, block, info)};
        const bool any_storage{(lhs && lhs->is_storage) || (rhs && rhs->is_storage)};
        if (!lhs || !rhs) {
            // Don't let the search fall through to the storage buffer address words
            return any_storage ? std::optional{MakeUntrackable()} : std::nullopt;
        }
        if (IsUntrackable(*lhs) || IsUntrackable(*rhs)) {
            return MakeUntrackable();
        }
        if (lhs->has_secondary || rhs->has_secondary) {
            return any_storage ? std::optional{MakeUntrackable()} : std::nullopt;
        }
        if (lhs->count > 1 || rhs->count > 1) {
            return any_storage ? std::optional{MakeUntrackable()} : std::nullopt;
        }
        if (rhs->is_storage && !lhs->is_storage) {
            // The storage sourced word is always the primary one
            std::swap(lhs, rhs);
        } else if (!lhs->is_storage &&
                   (lhs->shift_left > 0 || lhs->index > rhs->index || lhs->offset > rhs->offset)) {
            std::swap(lhs, rhs);
        }
        return ConstBufferAddr{
            .index = lhs->index,
            .offset = lhs->offset,
            .shift_left = lhs->shift_left,
            .secondary_index = rhs->index,
            .secondary_offset = rhs->offset,
            .secondary_shift_left = rhs->shift_left,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = true,
            .storage_offset = lhs->storage_offset,
            .storage_stride = lhs->storage_stride,
            .is_storage = lhs->is_storage,
            .secondary_storage_offset = rhs->storage_offset,
            .secondary_is_storage = rhs->is_storage,
        };
    }
    case IR::Opcode::ShiftLeftLogical32: {
        const IR::Value shift{inst->Arg(1)};
        if (!shift.IsImmediate()) {
            return std::nullopt;
        }
        std::optional lhs{TrackCached(inst->Arg(0), env, host_info, block, info)};
        if (lhs) {
            lhs->shift_left = shift.U32();
        }
        return lhs;
    }
    case IR::Opcode::BitwiseAnd32: {
        IR::Value op1{inst->Arg(0)};
        IR::Value op2{inst->Arg(1)};
        if (op1.IsImmediate()) {
            std::swap(op1, op2);
        }
        if (!op2.IsImmediate() && !op1.IsImmediate()) {
            do {
                auto try_index = TryGetConstant(op1, env);
                if (try_index) {
                    op1 = op2;
                    op2 = IR::Value{*try_index};
                    break;
                }
                auto try_index_2 = TryGetConstant(op2, env);
                if (try_index_2) {
                    op2 = IR::Value{*try_index_2};
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        std::optional lhs{TrackCached(op1, env, host_info, block, info)};
        if (lhs) {
            lhs->shift_left = static_cast<u32>(std::countr_zero(op2.U32()));
        }
        return lhs;
        break;
    }
    case IR::Opcode::GetCbufU32x2:
    case IR::Opcode::GetCbufU32:
        break;
    }
    const IR::Value index{inst->Arg(0)};
    const IR::Value offset{inst->Arg(1)};
    if (!index.IsImmediate()) {
        // Reading a bindless texture from variable indices is valid
        // but not supported here at the moment
        return std::nullopt;
    }
    if (offset.IsImmediate()) {
        return ConstBufferAddr{
            .index = index.U32(),
            .offset = offset.U32(),
            .shift_left = 0,
            .secondary_index = 0,
            .secondary_offset = 0,
            .secondary_shift_left = 0,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = false,
            .storage_offset = 0,
            .storage_stride = 0,
            .is_storage = false,
        };
    }
    IR::Inst* const offset_inst{offset.InstRecursive()};
    if (offset_inst->GetOpcode() != IR::Opcode::IAdd32) {
        return std::nullopt;
    }
    u32 base_offset{};
    IR::U32 dynamic_offset;
    if (offset_inst->Arg(0).IsImmediate()) {
        base_offset = offset_inst->Arg(0).U32();
        dynamic_offset = IR::U32{offset_inst->Arg(1)};
    } else if (offset_inst->Arg(1).IsImmediate()) {
        base_offset = offset_inst->Arg(1).U32();
        dynamic_offset = IR::U32{offset_inst->Arg(0)};
    } else {
        return std::nullopt;
    }
    auto const size_shift = DynamicDescriptorSizeShift(dynamic_offset);
    auto const sampled_limit = (std::max)(1U, (std::min)(host_info.max_per_stage_descriptor_sampled_images,
                                                         host_info.max_descriptor_set_sampled_images));
    auto const resource_limit = (std::max)(1U, host_info.max_per_stage_resources);
    return ConstBufferAddr{
        .index = index.U32(),
        .offset = base_offset,
        .shift_left = 0,
        .secondary_index = 0,
        .secondary_offset = 0,
        .secondary_shift_left = 0,
        .dynamic_offset = dynamic_offset,
        .count = DynamicDescriptorCount(base_offset, size_shift, (std::min)({DESCRIPTOR_MAX_COUNT, sampled_limit, resource_limit})),
        .has_secondary = false,
        .storage_offset = 0,
        .storage_stride = 0,
        .is_storage = false,
    };
}

bool IsAttributeSourcedHandle(const IR::Value& handle) {
    if (handle.IsImmediate()) {
        return false;
    }
    const IR::Inst* inst{handle.InstRecursive()};
    if (inst->GetOpcode() == IR::Opcode::BitCastU32F32 && !inst->Arg(0).IsImmediate()) {
        inst = inst->Arg(0).InstRecursive();
    }
    return inst->GetOpcode() == IR::Opcode::GetAttribute ||
           inst->GetOpcode() == IR::Opcode::GetAttributeU32;
}

TextureInst MakeInst(Environment& env, IR::Block* block, IR::Inst& inst, const HostTranslateInfo& host_info, const Info& info) {
    ConstBufferAddr addr;
    if (IsBindless(inst)) {
        const std::optional<ConstBufferAddr> track_addr{TrackCached(inst.Arg(0), env, host_info, block, info)};

        if (track_addr && IsUntrackable(*track_addr)) {
            // Better to fail loudly than to bind whatever the storage buffer address words decode
            // to (an out of range or garbage texture header)
            throw NotImplementedException(
                "Bindless texture handle loaded from global memory with an unsupported address");
        }
        if (track_addr) {
            addr = *track_addr;
        } else if (host_info.support_attribute_texture_handle &&
                   IsAttributeSourcedHandle(inst.Arg(0))) {
            // The handle travels with the vertices (sprite batches pick the texture per quad), so
            // there is no constant buffer to read it from. The host binds the handle of each run
            // of vertices that share one.
            addr = ConstBufferAddr{
                .index = ATTRIBUTE_HANDLE_CBUF_INDEX,
                .offset = 0,
                .shift_left = 0,
                .secondary_index = 0,
                .secondary_offset = 0,
                .secondary_shift_left = 0,
                .dynamic_offset = {},
                .count = 1,
                .has_secondary = false,
                .storage_offset = 0,
                .storage_stride = 0,
                .is_storage = false,
            };
        } else {
            throw NotImplementedException("Failed to track bindless texture constant buffer");
        }
    } else {
        addr = ConstBufferAddr{
            .index = env.TextureBoundBuffer(),
            .offset = inst.Arg(0).U32(),
            .shift_left = 0,
            .secondary_index = 0,
            .secondary_offset = 0,
            .secondary_shift_left = 0,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = false,
            .storage_offset = 0,
            .storage_stride = 0,
            .is_storage = false,
        };
    }
    return TextureInst{
        .cbuf = addr,
        .inst = &inst,
        .block = block,
    };
}

u32 GetTextureHandle(Environment& env, const ConstBufferAddr& cbuf) {
    return GetTextureHandleCached(env, cbuf);
}

[[maybe_unused]] TextureType ReadTextureType(Environment& env, const ConstBufferAddr& cbuf) {
    return env.ReadTextureType(GetTextureHandle(env, cbuf));
}

[[maybe_unused]] TexturePixelFormat ReadTexturePixelFormat(Environment& env, const ConstBufferAddr& cbuf) {
    return env.ReadTexturePixelFormat(GetTextureHandle(env, cbuf));
}

[[maybe_unused]] bool IsTexturePixelFormatInteger(Environment& env, const ConstBufferAddr& cbuf) {
    return env.IsTexturePixelFormatInteger(GetTextureHandle(env, cbuf));
}

class Descriptors {
public:
    explicit Descriptors(TextureBufferDescriptors& texture_buffer_descriptors_,
                         ImageBufferDescriptors& image_buffer_descriptors_,
                         TextureDescriptors& texture_descriptors_,
                         ImageDescriptors& image_descriptors_)
        : texture_buffer_descriptors{texture_buffer_descriptors_},
          image_buffer_descriptors{image_buffer_descriptors_},
          texture_descriptors{texture_descriptors_}, image_descriptors{image_descriptors_} {}

    u32 Add(const TextureBufferDescriptor& desc) {
        return Add(texture_buffer_descriptors, desc, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset &&
                   desc.shift_left == existing.shift_left &&
                   desc.secondary_cbuf_index == existing.secondary_cbuf_index &&
                   desc.secondary_cbuf_offset == existing.secondary_cbuf_offset &&
                   desc.secondary_shift_left == existing.secondary_shift_left &&
                   desc.count == existing.count && desc.size_shift == existing.size_shift &&
                   desc.has_secondary == existing.has_secondary &&
                   desc.storage_offset == existing.storage_offset &&
                   desc.storage_stride == existing.storage_stride &&
                   desc.is_storage == existing.is_storage &&
                   desc.secondary_storage_offset == existing.secondary_storage_offset &&
                   desc.secondary_is_storage == existing.secondary_is_storage;
        });
    }

    u32 Add(const ImageBufferDescriptor& desc) {
        const u32 index{Add(image_buffer_descriptors, desc, [&desc](const auto& existing) {
            return desc.format == existing.format && desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.size_shift == existing.size_shift &&
                   desc.storage_offset == existing.storage_offset &&
                   desc.storage_stride == existing.storage_stride &&
                   desc.is_storage == existing.is_storage;
        })};
        image_buffer_descriptors[index].is_written |= desc.is_written;
        image_buffer_descriptors[index].is_read |= desc.is_read;
        image_buffer_descriptors[index].is_integer |= desc.is_integer;
        return index;
    }

    u32 Add(const TextureDescriptor& desc) {
        const u32 index{Add(texture_descriptors, desc, [&desc](const auto& existing) {
            return desc.type == existing.type && desc.is_depth == existing.is_depth &&
                   desc.has_secondary == existing.has_secondary &&
                   desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset &&
                   desc.shift_left == existing.shift_left &&
                   desc.secondary_cbuf_index == existing.secondary_cbuf_index &&
                   desc.secondary_cbuf_offset == existing.secondary_cbuf_offset &&
                   desc.secondary_shift_left == existing.secondary_shift_left &&
                   desc.count == existing.count && desc.size_shift == existing.size_shift &&
                   desc.storage_offset == existing.storage_offset &&
                   desc.storage_stride == existing.storage_stride &&
                   desc.is_storage == existing.is_storage &&
                   desc.secondary_storage_offset == existing.secondary_storage_offset &&
                   desc.secondary_is_storage == existing.secondary_is_storage;
        })};
        // TODO: Read this from TIC
        texture_descriptors[index].is_multisample |= desc.is_multisample;
        texture_descriptors[index].is_integer |= desc.is_integer;
        return index;
    }

    u32 Add(const ImageDescriptor& desc) {
        const u32 index{Add(image_descriptors, desc, [&desc](const auto& existing) {
            return desc.type == existing.type && desc.format == existing.format &&
                   desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.size_shift == existing.size_shift &&
                   desc.storage_offset == existing.storage_offset &&
                   desc.storage_stride == existing.storage_stride &&
                   desc.is_storage == existing.is_storage;
        })};
        image_descriptors[index].is_written |= desc.is_written;
        image_descriptors[index].is_read |= desc.is_read;
        image_descriptors[index].is_integer |= desc.is_integer;
        return index;
    }

private:
    template <typename Descriptors, typename Descriptor, typename Func>
    static u32 Add(Descriptors& descriptors, const Descriptor& desc, Func&& pred) {
        // TODO: Handle arrays
        const auto it{std::ranges::find_if(descriptors, pred)};
        if (it != descriptors.end()) {
            return static_cast<u32>(std::distance(descriptors.begin(), it));
        }
        descriptors.push_back(desc);
        return static_cast<u32>(descriptors.size()) - 1;
    }

    TextureBufferDescriptors& texture_buffer_descriptors;
    ImageBufferDescriptors& image_buffer_descriptors;
    TextureDescriptors& texture_descriptors;
    ImageDescriptors& image_descriptors;
};

void PatchImageSampleImplicitLod(IR::Block& block, IR::Inst& inst) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    // Must run after the descriptor and array index of the sample are final, the size query reads
    // the same descriptor and array element
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const IR::Value coord(inst.Arg(1));
    const IR::Value index(inst.Arg(0));
    const IR::U32 lod{ir.Imm32(0)};
    const IR::U1 skip_mips{ir.Imm1(true)};
    const IR::Value texture_size = ir.ImageQueryDimension(index, lod, skip_mips, info);
    inst.SetArg(
        1, ir.CompositeConstruct(
               ir.FPMul(IR::F32(ir.CompositeExtract(coord, 0)),
                        ir.FPRecip(ir.ConvertUToF(32, 32, ir.CompositeExtract(texture_size, 0)))),
               ir.FPMul(IR::F32(ir.CompositeExtract(coord, 1)),
                        ir.FPRecip(ir.ConvertUToF(32, 32, ir.CompositeExtract(texture_size, 1))))));
}

bool IsPixelFormatSNorm(TexturePixelFormat pixel_format) {
    switch (pixel_format) {
    case TexturePixelFormat::A8B8G8R8_SNORM:
    case TexturePixelFormat::R8G8_SNORM:
    case TexturePixelFormat::R8_SNORM:
    case TexturePixelFormat::R16G16B16A16_SNORM:
    case TexturePixelFormat::R16G16_SNORM:
    case TexturePixelFormat::R16_SNORM:
        return true;
    default:
        return false;
    }
}

void PatchTexelFetch(IR::Block& block, IR::Inst& inst, TexturePixelFormat pixel_format) {
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    auto get_max_value = [pixel_format]() -> float {
        switch (pixel_format) {
        case TexturePixelFormat::A8B8G8R8_SNORM:
        case TexturePixelFormat::R8G8_SNORM:
        case TexturePixelFormat::R8_SNORM:
            return 1.f / (std::numeric_limits<char>::max)();
        case TexturePixelFormat::R16G16B16A16_SNORM:
        case TexturePixelFormat::R16G16_SNORM:
        case TexturePixelFormat::R16_SNORM:
            return 1.f / (std::numeric_limits<short>::max)();
        default:
            throw InvalidArgument("Invalid texture pixel format");
        }
    };

    const IR::Value new_inst{&*block.PrependNewInst(it, inst)};
    const IR::F32 x(ir.CompositeExtract(new_inst, 0));
    const IR::F32 y(ir.CompositeExtract(new_inst, 1));
    const IR::F32 z(ir.CompositeExtract(new_inst, 2));
    const IR::F32 w(ir.CompositeExtract(new_inst, 3));
    const IR::F16F32F64 max_value(ir.Imm32(get_max_value()));
    const IR::Value converted =
        ir.CompositeConstruct(ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(x)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(y)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(z)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(w)), max_value));
    inst.ReplaceUsesWith(converted);
}
} // Anonymous namespace

void TexturePass(Environment& env, IR::Program& program, const HostTranslateInfo& host_info) {
    // reset per-pass caches
    env.cbuf_word_cache.clear();
    env.handle_cache.clear();
    env.track_cache.clear();
    TextureInstVector to_replace;
    for (IR::Block* const block : program.post_order_blocks) {
        env.track_cache.clear();
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsTextureInstruction(inst)) {
                continue;
            }
            to_replace.push_back(MakeInst(env, block, inst, host_info, program.info));
        }
    }
    // Sort instructions to visit textures by constant buffer index, then by offset
    std::ranges::sort(to_replace, [](const auto& a, const auto& b) {
        if (a.cbuf.index != b.cbuf.index) return a.cbuf.index < b.cbuf.index;
        return a.cbuf.offset < b.cbuf.offset;
    });
    Descriptors descriptors{
        program.info.texture_buffer_descriptors,
        program.info.image_buffer_descriptors,
        program.info.texture_descriptors,
        program.info.image_descriptors,
    };
    const u32 sampled_dynamic_cap = DynamicSampledTextureCap(program.info, host_info, DynamicSampledTextureArrayCount(to_replace));
    bool has_last_is_integer{false};
    u32 last_handle{};
    bool last_is_integer{false};
    const auto is_texture_pixel_format_integer{[&](const ConstBufferAddr& cbuf_addr) {
        if (IsAttributeHandle(cbuf_addr)) {
            return false;
        }
        // Keyed by handle: descriptors read from the same storage buffer share (index, offset)
        const u32 handle{GetTextureHandleCached(env, cbuf_addr)};
        if (has_last_is_integer && last_handle == handle) {
            return last_is_integer;
        }
        last_is_integer = IsTexturePixelFormatIntegerCached(env, cbuf_addr);
        last_handle = handle;
        has_last_is_integer = true;
        return last_is_integer;
    }};
    for (TextureInst& texture_inst : to_replace) {
        // TODO: Handle arrays
        IR::Inst* const inst{texture_inst.inst};
        inst->ReplaceOpcode(IndexedInstruction(*inst));

        const auto& cbuf{texture_inst.cbuf};
        auto flags{inst->Flags<IR::TextureInstInfo>()};
        bool is_multisample{false};
        bool is_rect_sample{false};
        switch (inst->GetOpcode()) {
        case IR::Opcode::ImageQueryDimensions:
            flags.type.Assign(ReadTextureTypeCached(env, cbuf));
            inst->SetFlags(flags);
            break;
        case IR::Opcode::ImageSampleImplicitLod:
        case IR::Opcode::ImageSampleExplicitLod:
        case IR::Opcode::ImageSampleDrefImplicitLod:
        case IR::Opcode::ImageSampleDrefExplicitLod:
        case IR::Opcode::ImageGather:
        case IR::Opcode::ImageGatherDref:
            if (flags.type != TextureType::Color2D) {
                break;
            }
            is_rect_sample = ReadTextureTypeCached(env, cbuf) == TextureType::Color2DRect;
            break;
        case IR::Opcode::ImageFetch:
            if (flags.type == TextureType::Color2D || flags.type == TextureType::Color2DRect ||
                flags.type == TextureType::ColorArray2D) {
                is_multisample = !inst->Arg(4).IsEmpty();
            } else {
                inst->SetArg(4, IR::U32{});
            }
            if (flags.type != TextureType::Color1D) {
                break;
            }
            if (ReadTextureTypeCached(env, cbuf) == TextureType::Buffer) {
                // Replace with the bound texture type only when it's a texture buffer
                // If the instruction is 1D and the bound type is 2D, don't change the code and let
                // the rasterizer robustness handle it
                // This happens on Fire Emblem: Three Houses
                flags.type.Assign(TextureType::Buffer);
            }
            break;
        default:
            break;
        }
        u32 index;
        u32 size_shift = cbuf.count > 1 ? DynamicDescriptorSizeShift(cbuf.dynamic_offset) : DESCRIPTOR_SIZE_SHIFT;
        if (cbuf.is_storage) {
            // dynamic_offset already is the element index, storage_stride carries the stride
            size_shift = 0;
        }
        u32 count = cbuf.count;
        switch (inst->GetOpcode()) {
        case IR::Opcode::ImageRead:
        case IR::Opcode::ImageAtomicIAdd32:
        case IR::Opcode::ImageAtomicSMin32:
        case IR::Opcode::ImageAtomicUMin32:
        case IR::Opcode::ImageAtomicSMax32:
        case IR::Opcode::ImageAtomicUMax32:
        case IR::Opcode::ImageAtomicInc32:
        case IR::Opcode::ImageAtomicDec32:
        case IR::Opcode::ImageAtomicAnd32:
        case IR::Opcode::ImageAtomicOr32:
        case IR::Opcode::ImageAtomicXor32:
        case IR::Opcode::ImageAtomicExchange32:
        case IR::Opcode::ImageWrite: {
            // Images have no sampler, an immediate ORed into the handle is ignored
            if (cbuf.has_secondary && cbuf.secondary_index != IMMEDIATE_HANDLE_CBUF_INDEX) {
                throw NotImplementedException("Unexpected separate sampler");
            }
            const bool is_written{inst->GetOpcode() != IR::Opcode::ImageRead};
            const bool is_read{inst->GetOpcode() != IR::Opcode::ImageWrite};
            const bool is_integer{is_texture_pixel_format_integer(cbuf)};
            if (flags.type == TextureType::Buffer) {
                index = descriptors.Add(ImageBufferDescriptor{
                    .format = flags.image_format,
                    .is_written = is_written,
                    .is_read = is_read,
                    .is_integer = is_integer,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .count = count,
                    .size_shift = size_shift,
                    .storage_offset = cbuf.storage_offset,
                    .storage_stride = cbuf.storage_stride,
                    .is_storage = cbuf.is_storage,
                });
            } else {
                index = descriptors.Add(ImageDescriptor{
                    .type = flags.type,
                    .format = flags.image_format,
                    .is_written = is_written,
                    .is_read = is_read,
                    .is_integer = is_integer,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .count = count,
                    .size_shift = size_shift,
                    .storage_offset = cbuf.storage_offset,
                    .storage_stride = cbuf.storage_stride,
                    .is_storage = cbuf.is_storage,
                });
            }
            break;
        }
        default:
            if (flags.type == TextureType::Buffer) {
                index = descriptors.Add(TextureBufferDescriptor{
                    .has_secondary = cbuf.has_secondary,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .shift_left = cbuf.shift_left,
                    .secondary_cbuf_index = cbuf.secondary_index,
                    .secondary_cbuf_offset = cbuf.secondary_offset,
                    .secondary_shift_left = cbuf.secondary_shift_left,
                    .count = count,
                    .size_shift = size_shift,
                    .storage_offset = cbuf.storage_offset,
                    .storage_stride = cbuf.storage_stride,
                    .is_storage = cbuf.is_storage,
                    .secondary_storage_offset = cbuf.secondary_storage_offset,
                    .secondary_is_storage = cbuf.secondary_is_storage,
                });
            } else {
                count = std::min(count, sampled_dynamic_cap);
                const bool is_integer{is_texture_pixel_format_integer(cbuf)};
                index = descriptors.Add(TextureDescriptor{
                    .type = flags.type,
                    .is_depth = flags.is_depth != 0,
                    .is_multisample = is_multisample,
                    .is_integer = is_integer,
                    .has_secondary = cbuf.has_secondary,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .shift_left = cbuf.shift_left,
                    .secondary_cbuf_index = cbuf.secondary_index,
                    .secondary_cbuf_offset = cbuf.secondary_offset,
                    .secondary_shift_left = cbuf.secondary_shift_left,
                    .count = count,
                    .size_shift = size_shift,
                    .storage_offset = cbuf.storage_offset,
                    .storage_stride = cbuf.storage_stride,
                    .is_storage = cbuf.is_storage,
                    .secondary_storage_offset = cbuf.secondary_storage_offset,
                    .secondary_is_storage = cbuf.secondary_is_storage,
                });
                flags.is_integer.Assign(is_integer ? 1 : 0);
            }
            break;
        }
        flags.descriptor_index.Assign(index);
        inst->SetFlags(flags);

        if (count > 1) {
            const auto insert_point{IR::Block::InstructionList::s_iterator_to(*inst)};
            IR::IREmitter ir{*texture_inst.block, insert_point};
            const IR::U32 shift{ir.Imm32(size_shift)};
            inst->SetArg(0, ir.UMin(ir.ShiftRightLogical(cbuf.dynamic_offset, shift), ir.Imm32(count - 1)));
        } else {
            inst->SetArg(0, IR::Value{});
        }

        if (is_rect_sample) {
            PatchImageSampleImplicitLod(*texture_inst.block, *texture_inst.inst);
        }

        if (!host_info.support_snorm_render_buffer && inst->GetOpcode() == IR::Opcode::ImageFetch &&
            flags.type == TextureType::Buffer) {
            const auto pixel_format = ReadTexturePixelFormatCached(env, cbuf);
            if (IsPixelFormatSNorm(pixel_format)) {
                PatchTexelFetch(*texture_inst.block, *texture_inst.inst, pixel_format);
            }
        }
    }
}

void JoinTextureInfo(Info& base, Info& source) {
    Descriptors descriptors{
        base.texture_buffer_descriptors,
        base.image_buffer_descriptors,
        base.texture_descriptors,
        base.image_descriptors,
    };
    for (auto& desc : source.texture_buffer_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.image_buffer_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.texture_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.image_descriptors) {
        descriptors.Add(desc);
    }
}

} // namespace Shader::Optimization
