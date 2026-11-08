// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <fstream>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif
#include "common/heap_tracker.h"
#include "common/logging.h"
#include "common/assert.h"

namespace Common {

namespace {

// The budget starts 20000 maps below the limit, for other code and to account for split
// inaccuracy. Near it, the real map count decides: the budget grows while more than RaiseMargin
// maps are left, up to MinReserve below the limit, and shrinks when fewer than LowerMargin are
// left.
constexpr s64 DefaultReserve = 20000;
constexpr s64 MinReserve = 10000;
constexpr s64 RaiseMargin = 4000;
constexpr s64 LowerMargin = 1500;

// Counting the host mappings takes a while: at most this often while the budget moves, and less
// often once it has settled.
constexpr s64 BudgetCheckMs = 5000;
constexpr s64 SettledBudgetCheckMs = 30000;

// Each rebuild evicts this fraction of the resident mappings.
constexpr s64 EvictionDivisor = 16;

s64 GetMaxMapCount() {
    // Default value.
    s64 value = 65530;

    // Try to read how many mappings we can make.
    std::ifstream s("/proc/sys/vm/max_map_count");
    s >> value;

    // Print, for debug.
    LOG_INFO(HW_Memory, "Current maximum map count: {}", value);

    return value;
}

// Returns how many memory mappings the process has, or -1 if the host can't tell. This can run in
// the fault handler, so it doesn't allocate.
s64 CountHostMappings() {
#ifdef __linux__
    const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    s64 count = 0;
    std::array<char, 8 * 1024> buffer;
    while (true) {
        const ssize_t read_size = read(fd, buffer.data(), buffer.size());
        if (read_size > 0) {
            count += std::count(buffer.data(), buffer.data() + read_size, '\n');
        } else if (read_size == 0 || errno != EINTR) {
            if (read_size < 0) {
                count = -1;
            }
            break;
        }
    }
    close(fd);
    return count;
#else
    return -1;
#endif
}

} // namespace

HeapTracker::HeapTracker(Common::HostMemory& buffer)
    : m_buffer(buffer), m_map_count_limit(GetMaxMapCount()),
      m_min_budget(std::max<s64>(m_map_count_limit - DefaultReserve, 0) / 2),
      m_max_resident_map_count(std::max<s64>(m_map_count_limit - DefaultReserve, 0)),
      m_budget_ceiling(std::max<s64>(m_map_count_limit - MinReserve, 0)) {}
HeapTracker::~HeapTracker() = default;

bool HeapTracker::UpdateBudget() {
    // One core at a time, and not too often.
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 next = m_next_budget_update.load(std::memory_order_relaxed);
    if (now < next || !m_next_budget_update.compare_exchange_strong(next, now + BudgetCheckMs)) {
        return false;
    }
    const s64 host_mappings = CountHostMappings();
    if (host_mappings < 0) {
        return false;
    }

    std::scoped_lock lk{m_lock};
    const s64 left = m_map_count_limit - host_mappings;
    s64 budget = m_max_resident_map_count;
    if (left > RaiseMargin) {
        // A new resident mapping adds up to two host mappings when it splits a free range.
        budget = (std::max)(budget, (std::min)(m_resident_map_count + (left - RaiseMargin) / 2,
                                               m_budget_ceiling));
    } else if (left < LowerMargin) {
        budget = (std::max)(m_min_budget,
                            (std::min)(budget, m_resident_map_count) - (LowerMargin - left));
    }
    if (budget != m_max_resident_map_count) {
        LOG_INFO(HW_Memory, "Separate heap budget {} -> {} ({} host mappings)",
                 m_max_resident_map_count, budget, host_mappings);
        m_max_resident_map_count = budget;
    } else if (budget >= m_map_count_limit - MinReserve ||
               left - 2 * (std::max)(budget - m_resident_map_count, s64{0}) >= 2 * RaiseMargin) {
        // Settled, and a full budget leaves room to spare: look again less often.
        m_next_budget_update.store(now + SettledBudgetCheckMs, std::memory_order_relaxed);
    }
    return m_resident_map_count <= m_max_resident_map_count;
}

void HeapTracker::Map(size_t virtual_offset, size_t host_offset, size_t length,
                      MemoryPermission perm, bool is_separate_heap) {
    // When mapping other memory, map pages immediately.
    if (!is_separate_heap) {
        m_buffer.Map(virtual_offset, host_offset, length, perm, false);
        return;
    }

    {
        // We are mapping part of a separate heap.
        std::scoped_lock lk{m_lock};

        auto* const map = new SeparateHeapMap{
            .vaddr = virtual_offset,
            .paddr = host_offset,
            .size = length,
            .tick = m_tick++,
            .perm = perm,
            .is_resident = false,
        };

        // Insert into mappings.
        m_map_count++;
        m_mappings.insert(*map);
    }

    // Finally, map.
    this->DeferredMapSeparateHeap(virtual_offset);
}

void HeapTracker::Unmap(size_t virtual_offset, size_t size, bool is_separate_heap) {
    // If this is a separate heap...
    if (is_separate_heap) {
        std::scoped_lock lk{m_lock};

        const SeparateHeapMap key{
            .vaddr = virtual_offset,
        };

        // Split at the boundaries of the region we are removing.
        this->SplitHeapMapLocked(virtual_offset);
        this->SplitHeapMapLocked(virtual_offset + size);

        // Erase all mappings in range.
        auto it = m_mappings.find(key);
        while (it != m_mappings.end() && it->vaddr < virtual_offset + size) {
            // Get underlying item.
            auto* const item = std::addressof(*it);

            // If resident, erase from resident map.
            if (item->is_resident) {
                ASSERT(--m_resident_map_count >= 0);
                m_resident_mappings.erase(m_resident_mappings.iterator_to(*item));
            }

            // Erase from map.
            ASSERT(--m_map_count >= 0);
            it = m_mappings.erase(it);

            // Free the item.
            delete item;
        }
    }

    // Unmap pages.
    m_buffer.Unmap(virtual_offset, size, false);
}

void HeapTracker::Protect(size_t virtual_offset, size_t size, MemoryPermission perm) {
    // Ensure no rebuild occurs while reprotecting.
    std::shared_lock lk{m_rebuild_lock};

    // Split at the boundaries of the region we are reprotecting.
    this->SplitHeapMap(virtual_offset, size);

    // Declare tracking variables.
    const VAddr end = virtual_offset + size;
    VAddr cur = virtual_offset;

    while (cur < end) {
        VAddr next = cur;
        bool should_protect = false;

        {
            std::scoped_lock lk2{m_lock};

            const SeparateHeapMap key{
                .vaddr = next,
            };

            // Try to get the next mapping corresponding to this address.
            const auto it = m_mappings.nfind(key);

            if (it == m_mappings.end()) {
                // There are no separate heap mappings remaining.
                next = end;
                should_protect = true;
            } else if (it->vaddr == cur) {
                // We are in range.
                // Update permission bits.
                it->perm = perm;

                // Determine next address and whether we should protect.
                next = cur + it->size;
                should_protect = it->is_resident;
            } else /* if (it->vaddr > cur) */ {
                // We weren't in range, but there is a block coming up that will be.
                next = it->vaddr;
                should_protect = true;
            }
        }

        // Clamp to end.
        next = (std::min)(next, end);
        // Reprotect, if we need to.
        if (should_protect) {
            m_buffer.Protect(cur, next - cur, perm);
        }

        // Advance.
        cur = next;
    }
}

bool HeapTracker::DeferredMapSeparateHeap(u8* fault_address) {
    if (m_buffer.IsInVirtualRange(fault_address)) {
        return this->DeferredMapSeparateHeap(fault_address - m_buffer.VirtualBasePointer());
    }

    return false;
}

bool HeapTracker::DeferredMapSeparateHeap(size_t virtual_offset) {
    // If the host is out of mappings, evict and try once more.
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool rebuild_required = false;
        bool near_budget = false;
        bool mapped = false;
        std::size_t evictions_seen = 0;

        {
            std::scoped_lock lk{m_lock};

            // Check to ensure this was a non-resident separate heap mapping.
            const auto it = this->GetNearestHeapMapLocked(virtual_offset);
            if (it == m_mappings.end() || it->is_resident) {
                return false;
            }

            // Update tick before possible rebuild.
            it->tick = m_tick++;

            // Check if we need to rebuild.
            if (m_resident_map_count > m_max_resident_map_count) {
                rebuild_required = true;
            }
            near_budget = m_resident_map_count >=
                          m_max_resident_map_count - m_max_resident_map_count / EvictionDivisor;
            evictions_seen = m_evictions;

            // Map the area.
            mapped = m_buffer.TryMap(it->vaddr, it->paddr, it->size, it->perm, false);
            if (mapped) {
                // This map is now resident.
                it->is_resident = true;
                m_resident_map_count++;
                m_resident_mappings.insert(*it);
            } else if (attempt == 0 && m_resident_map_count > 0) {
                // Out of host mappings below the budget: never aim that high again.
                const s64 lowered = (std::max)(
                    m_resident_map_count - m_resident_map_count / EvictionDivisor, m_min_budget);
                if (lowered < m_max_resident_map_count) {
                    LOG_WARNING(HW_Memory, "Out of host memory mappings, lowering budget to {}",
                                lowered);
                    m_max_resident_map_count = lowered;
                }
                m_budget_ceiling = (std::min)(m_budget_ceiling, lowered);
            }
        }

        if (mapped) {
            // Close to the budget, recount the host mappings: if there is room for a bigger budget,
            // no eviction is needed. Otherwise the eviction happens now if it was due, or on a later
            // fault.
            if ((rebuild_required || near_budget) && this->UpdateBudget()) {
                rebuild_required = false;
            }
            if (rebuild_required) {
                this->RebuildSeparateHeapAddressSpace(false, 0);
            }
            return true;
        }

        if (attempt == 1 || !this->RebuildSeparateHeapAddressSpace(true, evictions_seen)) {
            break;
        }
    }

    // Only other code freeing mappings can help now.
    std::scoped_lock lk{m_lock};
    if (!m_out_of_mappings_logged) {
        LOG_ERROR(HW_Memory, "Out of host memory mappings, the separate heap can't be mapped");
        m_out_of_mappings_logged = true;
    }
    return true;
}

bool HeapTracker::RebuildSeparateHeapAddressSpace(bool force, std::size_t evictions_seen) {
    std::scoped_lock lk{m_rebuild_lock, m_lock};

    // Another core may have made room, or raised the budget, in the meantime.
    if (force && m_evictions != evictions_seen) {
        return true;
    }
    if (m_resident_mappings.empty() ||
        (!force && m_resident_map_count <= m_max_resident_map_count)) {
        return false;
    }

    // Dump the oldest sixteenth of the mappings, or one sixteenth below a budget lowered after
    // running out of host mappings. Dumped mappings that are still in use fault back in right away,
    // and dumping half of them at once can stall every guest core for a second.
    const s64 slice = (std::max)(m_resident_map_count / EvictionDivisor, s64{1});
    std::size_t const desired_count =
        force ? (std::min)(m_resident_map_count - slice, m_max_resident_map_count)
              : (std::min)(m_resident_map_count, m_max_resident_map_count) *
                    (EvictionDivisor - 1) / EvictionDivisor;
    std::size_t const evict_count = m_resident_map_count - desired_count;
    auto it = m_resident_mappings.begin();
    std::size_t evicted = 0;
    int failures = 0;

    for (size_t i = 0; i < evict_count && it != m_resident_mappings.end(); i++) {
        // Unmap and unmark.
        if (!m_buffer.TryUnmap(it->vaddr, it->size, false)) {
            // The host can't take it back right now: it stays resident, at the back of the queue.
            // Past its limit the host refuses them all, so don't insist.
            auto& map = *it;
            it = m_resident_mappings.erase(it);
            map.tick = m_tick++;
            m_resident_mappings.insert(map);
            if (++failures == 8) {
                break;
            }
            continue;
        }
        it->is_resident = false;

        // Advance.
        ASSERT(--m_resident_map_count >= 0);
        it = m_resident_mappings.erase(it);
        evicted++;
    }

    if (evicted == 0) {
        return false;
    }
    m_evictions++;
    return true;
}

void HeapTracker::SplitHeapMap(VAddr offset, size_t size) {
    std::scoped_lock lk{m_lock};

    this->SplitHeapMapLocked(offset);
    this->SplitHeapMapLocked(offset + size);
}

void HeapTracker::SplitHeapMapLocked(VAddr offset) {
    const auto it = this->GetNearestHeapMapLocked(offset);
    if (it == m_mappings.end() || it->vaddr == offset) {
        // Not contained or no split required.
        return;
    }

    // Cache the original values.
    auto* const left = std::addressof(*it);
    const size_t orig_size = left->size;

    // Adjust the left map.
    const size_t left_size = offset - left->vaddr;
    left->size = left_size;

    // Create the new right map.
    auto* const right = new SeparateHeapMap{
        .vaddr = left->vaddr + left_size,
        .paddr = left->paddr + left_size,
        .size = orig_size - left_size,
        .tick = left->tick,
        .perm = left->perm,
        .is_resident = left->is_resident,
    };

    // Insert the new right map.
    m_map_count++;
    m_mappings.insert(*right);

    // If resident, also insert into resident map.
    if (right->is_resident) {
        m_resident_map_count++;
        m_resident_mappings.insert(*right);
    }
}

HeapTracker::AddrTree::iterator HeapTracker::GetNearestHeapMapLocked(VAddr offset) {
    const SeparateHeapMap key{
        .vaddr = offset,
    };

    return m_mappings.find(key);
}

} // namespace Common
