// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <new>

#import <QuartzCore/QuartzCore.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_metal/mtl_buffer_cache.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_texture_cache.h"

namespace Metal {

namespace {

using VideoCore::Surface::PixelFormat;

/// Metal blit copies and fills require 4 byte aligned offsets and sizes on macOS.
constexpr u64 BLIT_ALIGNMENT = 4;

/// Constant buffer offsets must be 256 byte aligned on non-Apple GPUs, keep the stream buffer
/// valid for both.
constexpr size_t STREAM_UNIFORM_ALIGNMENT = 256;

bool IsBlitAligned(u64 src_offset, u64 dst_offset, u64 size) {
    return (src_offset | dst_offset | size) % BLIT_ALIGNMENT == 0;
}

u32 IndexSize(Tegra::Engines::Maxwell3D::Regs::IndexFormat format) {
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;
    switch (format) {
    case IndexFormat::UnsignedByte:
        return 1;
    case IndexFormat::UnsignedShort:
        return 2;
    case IndexFormat::UnsignedInt:
        return 4;
    }
    ASSERT_MSG(false, "Invalid index format={}", format);
    return 4;
}

u32 ReadIndex(const u8* data, u32 index_size, u32 index) {
    switch (index_size) {
    case 1:
        return data[index];
    case 2: {
        u16 value;
        std::memcpy(&value, data + index * 2, sizeof(value));
        return value;
    }
    default: {
        u32 value;
        std::memcpy(&value, data + static_cast<size_t>(index) * 4, sizeof(value));
        return value;
    }
    }
}

/// The texture a view was created from, or the texture itself.
id<MTLTexture> RootTexture(id<MTLTexture> texture) {
    while (texture.parentTexture) {
        texture = texture.parentTexture;
    }
    return texture;
}

/// Emits two triangles per quad, a quad being (v0, v1, v2, v3) in drawing order.
template <typename Func>
void ForEachQuadTriangleVertex(bool is_strip, u32 count, Func&& emit) {
    if (is_strip) {
        if (count < 4) {
            return;
        }
        const u32 num_quads = (count - 2) / 2;
        for (u32 quad = 0; quad < num_quads; ++quad) {
            const u32 v = quad * 2;
            for (const u32 offset : {0U, 1U, 3U, 0U, 3U, 2U}) {
                emit(v + offset);
            }
        }
    } else {
        const u32 num_quads = count / 4;
        for (u32 quad = 0; quad < num_quads; ++quad) {
            const u32 v = quad * 4;
            for (const u32 offset : {0U, 1U, 2U, 0U, 2U, 3U}) {
                emit(v + offset);
            }
        }
    }
}

} // Anonymous namespace

Scheduler::Scheduler(const Device& device)
    : queue{device.GetCommandQueue()},
      pending_callbacks{std::make_shared<std::vector<std::function<void()>>>()} {}

Scheduler::~Scheduler() {
    Finish();
}

id<MTLCommandBuffer> Scheduler::CommandBuffer() {
    if (!command_buffer) {
        // The queue returns autoreleased objects, the member keeps them alive past the pool.
        @autoreleasepool {
            command_buffer = [queue commandBuffer];
            command_buffer.label = @"Eden cache commands";
        }
    }
    return command_buffer;
}

id<MTLBlitCommandEncoder> Scheduler::BlitEncoder(PassEnd reason) {
    if (!blit_encoder) {
        // Only one encoder can be open on a command buffer.
        EndEncoding(reason);
        @autoreleasepool {
            blit_encoder = [CommandBuffer() blitCommandEncoder];
        }
        ++blit_generation;
        ++stats.blit_encoders;
    }
    return blit_encoder;
}

id<MTLBlitCommandEncoder> Scheduler::BlitEncoderFor(u64& first_tag, u64& second_tag) {
    if (blit_encoder && (first_tag == blit_generation || second_tag == blit_generation)) {
        EndBlitEncoding();
    }
    id<MTLBlitCommandEncoder> encoder = BlitEncoder(PassEnd::ImageBlit);
    first_tag = blit_generation;
    second_tag = blit_generation;
    return encoder;
}

id<MTLCommandBuffer> Scheduler::SideCommandBuffer() {
    if (!side_command_buffer) {
        @autoreleasepool {
            side_command_buffer = [queue commandBuffer];
            side_command_buffer.label = @"Eden side uploads";
        }
    }
    return side_command_buffer;
}

id<MTLBlitCommandEncoder> Scheduler::SideBlitEncoder() {
    if (!side_blit_encoder) {
        @autoreleasepool {
            side_blit_encoder = [SideCommandBuffer() blitCommandEncoder];
        }
    }
    return side_blit_encoder;
}

void Scheduler::SideCopy(id<MTLBuffer> src, u64 src_offset, id<MTLBuffer> dst, u64 dst_offset,
                         u64 size) {
    if (size == 0) {
        return;
    }
    // Created now so that IsEmpty() reports the pending copies until they are submitted.
    (void)SideCommandBuffer();
    if (!side_copies.empty()) {
        SideCopyCommand& last = side_copies.back();
        if (last.src == src && last.dst == dst && last.src_offset + last.size == src_offset &&
            last.dst_offset + last.size == dst_offset) {
            last.size += size;
            return;
        }
    }
    side_copies.push_back(SideCopyCommand{
        .src = src,
        .dst = dst,
        .src_offset = src_offset,
        .dst_offset = dst_offset,
        .size = size,
    });
}

void Scheduler::EncodeSideCopies() {
    if (side_copies.empty()) {
        return;
    }
    id<MTLBlitCommandEncoder> encoder = SideBlitEncoder();
    for (const SideCopyCommand& copy : side_copies) {
        [encoder copyFromBuffer:copy.src
                   sourceOffset:copy.src_offset
                       toBuffer:copy.dst
              destinationOffset:copy.dst_offset
                           size:copy.size];
    }
    stats.side_blits += side_copies.size();
    side_copies.clear();
}

void Scheduler::MarkTextureUsed(id<MTLTexture> texture) {
    if (!texture) {
        return;
    }
    const id<MTLTexture> root = RootTexture(texture);
    for (id<MTLTexture> used : tick_textures) {
        if (used == root) {
            return;
        }
    }
    tick_textures.push_back(root);
}

bool Scheduler::IsTextureUsedThisTick(id<MTLTexture> texture) const {
    if (!texture || tick_textures.empty()) {
        return false;
    }
    const id<MTLTexture> root = RootTexture(texture);
    for (id<MTLTexture> used : tick_textures) {
        if (used == root) {
            return true;
        }
    }
    return false;
}

id<MTLRenderCommandEncoder> Scheduler::RenderEncoder(MTLRenderPassDescriptor* key,
                                                     MTLRenderPassDescriptor* descriptor) {
    if (render_encoder && render_pass_key == key && !descriptor) {
        return render_encoder;
    }
    EndEncoding();
    bool clears_depth = false;
    bool clears_stencil = false;
    if (!pending_clears.empty()) {
        if (!descriptor) {
            // A clear deferred on this framebuffer becomes the load actions of the pass.
            for (auto it = pending_clears.begin(); it != pending_clears.end(); ++it) {
                if (it->key != key) {
                    continue;
                }
                descriptor = it->descriptor;
                clears_depth = it->clears_depth;
                clears_stencil = it->clears_stencil;
                pending_clears.erase(it);
                ++stats.applied_clears;
                break;
            }
        }
        FlushPendingClearsOverlapping(descriptor ? descriptor : key);
    }
    if (key != nil && last_ended_pass_key == key) {
        CountRestart(last_pass_end);
    }
    last_ended_pass_key = nil;
    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = descriptor ? descriptor : key;
        render_encoder = [CommandBuffer() renderCommandEncoderWithDescriptor:pass];
        if (render_encoder) {
            // Store actions the framebuffer left unknown are finalized when the pass ends. An
            // aspect cleared by its load action holds new data whatever the draws do.
            MTLRenderPassDepthAttachmentDescriptor* depth = pass.depthAttachment;
            MTLRenderPassStencilAttachmentDescriptor* stencil = pass.stencilAttachment;
            depth_store_pending =
                depth.texture != nil && depth.storeAction == MTLStoreActionUnknown;
            stencil_store_pending =
                stencil.texture != nil && stencil.storeAction == MTLStoreActionUnknown;
            if (depth.texture != nil && depth.loadAction == MTLLoadActionClear) {
                clears_depth = true;
            }
            if (stencil.texture != nil && stencil.loadAction == MTLLoadActionClear) {
                clears_stencil = true;
            }
        }
    }
    if (!render_encoder) {
        LOG_ERROR(Render_Metal, "Failed to create a render command encoder");
        return nil;
    }
    render_pass_key = key;
    ++encoder_generation;
    ++stats.render_passes;
    if (clears_depth) {
        depth_write_generation = encoder_generation;
    }
    if (clears_stencil) {
        stencil_write_generation = encoder_generation;
    }
    return render_encoder;
}

id<MTLComputeCommandEncoder> Scheduler::ComputeEncoder() {
    if (!compute_encoder) {
        EndEncoding(PassEnd::Compute);
        @autoreleasepool {
            compute_encoder = [CommandBuffer() computeCommandEncoder];
        }
        ++encoder_generation;
        ++stats.compute_encoders;
    }
    return compute_encoder;
}

void Scheduler::CountRestart(PassEnd reason) noexcept {
    ++stats.pass_restarts;
    switch (reason) {
    case PassEnd::BufferBlit:
        ++stats.restarts_buffer_blit;
        break;
    case PassEnd::ImageBlit:
        ++stats.restarts_image_blit;
        break;
    case PassEnd::Compute:
        ++stats.restarts_compute;
        break;
    case PassEnd::Clear:
        ++stats.restarts_clear;
        break;
    case PassEnd::BlitPass:
        ++stats.restarts_blit_pass;
        break;
    case PassEnd::Barrier:
        ++stats.restarts_barrier;
        break;
    case PassEnd::FeedbackLoop:
        ++stats.restarts_feedback_loop;
        break;
    case PassEnd::Flush:
        ++stats.restarts_flush;
        break;
    case PassEnd::Other:
        ++stats.restarts_other;
        break;
    }
}

void Scheduler::EndEncoding(PassEnd reason) {
    if (blit_encoder) {
        [blit_encoder endEncoding];
        blit_encoder = nil;
    }
    bool ended_shader_encoder = false;
    if (render_encoder) {
        FinalizeStoreActions();
        [render_encoder endEncoding];
        render_encoder = nil;
        last_ended_pass_key = render_pass_key;
        last_pass_end = reason;
        render_pass_key = nil;
        ended_shader_encoder = true;
    }
    if (compute_encoder) {
        [compute_encoder endEncoding];
        compute_encoder = nil;
        ended_shader_encoder = true;
    }
    if (ended_shader_encoder) {
        RunEncoderEndCallback();
    }
}

void Scheduler::FinalizeStoreActions() {
    // Tile based GPUs write a stored attachment back to memory from tile memory when the pass
    // ends. An aspect that was only loaded and tested still holds what memory holds.
    if (depth_store_pending) {
        const bool wrote = PassWroteDepth();
        [render_encoder setDepthStoreAction:wrote ? MTLStoreActionStore : MTLStoreActionDontCare];
        if (!wrote) {
            ++stats.depth_dontcare_stores;
        }
        depth_store_pending = false;
    }
    if (stencil_store_pending) {
        const bool wrote = PassWroteStencil();
        [render_encoder
            setStencilStoreAction:wrote ? MTLStoreActionStore : MTLStoreActionDontCare];
        if (!wrote) {
            ++stats.stencil_dontcare_stores;
        }
        stencil_store_pending = false;
    }
}

void Scheduler::RunEncoderEndCallback() {
    if (!on_encoder_end || in_encoder_end_callback) {
        return;
    }
    // Deferred work is encoded now, between the encoder and whatever comes next. It may open
    // a blit encoder, which the caller expects to be closed as well.
    in_encoder_end_callback = true;
    on_encoder_end();
    in_encoder_end_callback = false;
    if (blit_encoder) {
        [blit_encoder endEncoding];
        blit_encoder = nil;
    }
}

void Scheduler::EndBlitEncoding() {
    if (blit_encoder) {
        [blit_encoder endEncoding];
        blit_encoder = nil;
    }
}

void Scheduler::Flush() {
    EndEncoding(PassEnd::Flush);
    // Deferred uploads recorded while no encoder was open belong to this submission too:
    // their staging memory is recycled once its tick completes.
    RunEncoderEndCallback();
    const u64 tick = current_tick;
    auto callbacks = std::move(pending_callbacks);
    pending_callbacks = std::make_shared<std::vector<std::function<void()>>>();

    @autoreleasepool {
        // The side uploads go first: the queue runs command buffers in commit order.
        if (side_command_buffer) {
            EncodeSideCopies();
            if (side_blit_encoder) {
                [side_blit_encoder endEncoding];
                side_blit_encoder = nil;
            }
            [side_command_buffer commit];
            side_command_buffer = nil;
            ++stats.side_command_buffers;
        }
        id<MTLCommandBuffer> submit = CommandBuffer();
        if (!submit) {
            LOG_CRITICAL(Render_Metal, "Failed to create a command buffer");
            for (auto& callback : *callbacks) {
                callback();
            }
            SignalTick(tick);
        } else {
            Scheduler* const self = this;
            const CFTimeInterval commit_time = CACurrentMediaTime();
            [submit addCompletedHandler:^(id<MTLCommandBuffer> completed) {
              if (completed.status == MTLCommandBufferStatusError) {
                  NSString* const reason = completed.error.localizedDescription;
                  LOG_ERROR(Render_Metal, "Cache command buffer failed: {}",
                            reason ? reason.UTF8String : "unknown error");
              }
              // GPU timestamps share the host time base of CACurrentMediaTime. Consecutive
              // command buffers overlap on the GPU (the next one starts while the last passes
              // of the previous one finish), so only the time past the end of what was already
              // counted is busy time.
              CFTimeInterval busy = 0.0;
              {
                  std::scoped_lock time_lock{self->gpu_time_mutex};
                  const CFTimeInterval start =
                      (std::max)(completed.GPUStartTime, self->gpu_timeline_end);
                  if (completed.GPUEndTime > start) {
                      busy = completed.GPUEndTime - start;
                  }
                  self->gpu_timeline_end = (std::max)(self->gpu_timeline_end, completed.GPUEndTime);
              }
              const CFTimeInterval lag = completed.GPUStartTime - commit_time;
              if (busy > 0.0) {
                  self->gpu_busy_ns.fetch_add(static_cast<u64>(busy * 1e9),
                                              std::memory_order_relaxed);
              }
              if (lag > 0.0) {
                  self->gpu_lag_ns.fetch_add(static_cast<u64>(lag * 1e9),
                                             std::memory_order_relaxed);
              }
              self->gpu_timed_buffers.fetch_add(1, std::memory_order_relaxed);
              for (auto& callback : *callbacks) {
                  callback();
              }
              self->SignalTick(tick);
            }];
            [submit commit];
            ++stats.command_buffers;
        }
    }
    command_buffer = nil;
    tick_textures.clear();
    ++current_tick;
}

void Scheduler::Finish() {
    FlushPendingClears(ClearFlushReason::Flush);
    Flush();
    Wait(current_tick - 1);
}

void Scheduler::Wait(u64 tick, WaitReason reason) {
    if (tick >= current_tick) {
        if (IsEmpty()) {
            // Nothing was recorded since the last submission, so it holds everything.
            tick = current_tick - 1;
        } else {
            Flush();
        }
    }
    if (IsFree(tick)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    {
        std::unique_lock lock{tick_mutex};
        tick_cv.wait(lock, [this, tick] { return IsFree(tick); });
    }
    ++stats.gpu_waits;
    stats.gpu_wait_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - start)
                                              .count());
    switch (reason) {
    case WaitReason::BufferDownload:
        ++stats.waits_buffer_download;
        break;
    case WaitReason::CpuRead:
        ++stats.waits_cpu_read;
        break;
    case WaitReason::ImageDownload:
        ++stats.waits_image_download;
        break;
    case WaitReason::UnalignedCopy:
        ++stats.waits_unaligned_copy;
        break;
    case WaitReason::Sync:
        ++stats.waits_sync;
        break;
    }
}

void Scheduler::DeferClear(MTLRenderPassDescriptor* key, MTLRenderPassDescriptor* descriptor,
                           bool clears_depth, bool clears_stencil) {
    ++stats.deferred_clears;
    for (PendingClear& pending : pending_clears) {
        if (pending.key != key) {
            continue;
        }
        // Nothing touched the framebuffer since the earlier clear, or it would have run: the
        // attachments it cleared that this one doesn't are carried over.
        @autoreleasepool {
            for (NSUInteger index = 0; index < MAX_PASS_TEXTURES - 2; ++index) {
                MTLRenderPassColorAttachmentDescriptor* earlier =
                    pending.descriptor.colorAttachments[index];
                MTLRenderPassColorAttachmentDescriptor* current = descriptor.colorAttachments[index];
                if (earlier.texture && earlier.loadAction == MTLLoadActionClear &&
                    current.loadAction != MTLLoadActionClear) {
                    current.loadAction = MTLLoadActionClear;
                    current.clearColor = earlier.clearColor;
                }
            }
            MTLRenderPassDepthAttachmentDescriptor* earlier_depth = pending.descriptor.depthAttachment;
            MTLRenderPassDepthAttachmentDescriptor* depth = descriptor.depthAttachment;
            if (earlier_depth.texture && earlier_depth.loadAction == MTLLoadActionClear &&
                depth.loadAction != MTLLoadActionClear) {
                depth.loadAction = MTLLoadActionClear;
                depth.clearDepth = earlier_depth.clearDepth;
            }
            MTLRenderPassStencilAttachmentDescriptor* earlier_stencil =
                pending.descriptor.stencilAttachment;
            MTLRenderPassStencilAttachmentDescriptor* stencil = descriptor.stencilAttachment;
            if (earlier_stencil.texture && earlier_stencil.loadAction == MTLLoadActionClear &&
                stencil.loadAction != MTLLoadActionClear) {
                stencil.loadAction = MTLLoadActionClear;
                stencil.clearStencil = earlier_stencil.clearStencil;
            }
        }
        pending.descriptor = descriptor;
        pending.clears_depth |= clears_depth;
        pending.clears_stencil |= clears_stencil;
        return;
    }
    if (pending_clears.size() >= MAX_PENDING_CLEARS) {
        const PendingClear oldest = std::move(pending_clears.front());
        pending_clears.erase(pending_clears.begin());
        ExecutePendingClear(oldest, ClearFlushReason::Evicted);
    }
    PendingClear clear{
        .key = key,
        .descriptor = descriptor,
        .clears_depth = clears_depth,
        .clears_stencil = clears_stencil,
    };
    clear.num_textures = PassTextures(descriptor, clear.textures);
    pending_clears.push_back(std::move(clear));
}

bool Scheduler::HasPendingClearFor(id<MTLTexture> texture) const {
    if (pending_clears.empty() || !texture) {
        return false;
    }
    const id<MTLTexture> root = RootTexture(texture);
    for (const PendingClear& pending : pending_clears) {
        for (size_t i = 0; i < pending.num_textures; ++i) {
            if (pending.textures[i] == root) {
                return true;
            }
        }
    }
    return false;
}

void Scheduler::FlushPendingClears(ClearFlushReason reason) {
    if (pending_clears.empty()) {
        return;
    }
    const std::vector<PendingClear> clears = std::move(pending_clears);
    pending_clears.clear();
    for (const PendingClear& clear : clears) {
        ExecutePendingClear(clear, reason);
    }
}

void Scheduler::FlushPendingClearsFor(std::initializer_list<id<MTLTexture>> textures,
                                      ClearFlushReason reason) {
    if (pending_clears.empty()) {
        return;
    }
    std::array<id<MTLTexture>, MAX_PASS_TEXTURES> roots;
    size_t num_roots = 0;
    for (id<MTLTexture> texture : textures) {
        if (texture && num_roots < roots.size()) {
            roots[num_roots++] = RootTexture(texture);
        }
    }
    FlushPendingClearsMatching(std::span<const id<MTLTexture>>{roots.data(), num_roots}, reason);
}

void Scheduler::FlushPendingClearsOverlapping(MTLRenderPassDescriptor* descriptor) {
    std::array<id<MTLTexture>, MAX_PASS_TEXTURES> roots;
    const size_t num_roots = PassTextures(descriptor, roots);
    FlushPendingClearsMatching(std::span<const id<MTLTexture>>{roots.data(), num_roots},
                               ClearFlushReason::PassOverlap);
}

void Scheduler::FlushPendingClearsMatching(std::span<const id<MTLTexture>> roots,
                                           ClearFlushReason reason) {
    if (pending_clears.empty() || roots.empty()) {
        return;
    }
    std::vector<PendingClear> matching;
    for (auto it = pending_clears.begin(); it != pending_clears.end();) {
        bool matches = false;
        for (size_t i = 0; i < it->num_textures && !matches; ++i) {
            for (id<MTLTexture> root : roots) {
                if (it->textures[i] == root) {
                    matches = true;
                    break;
                }
            }
        }
        if (matches) {
            matching.push_back(std::move(*it));
            it = pending_clears.erase(it);
        } else {
            ++it;
        }
    }
    for (const PendingClear& clear : matching) {
        ExecutePendingClear(clear, reason);
    }
}

void Scheduler::ExecutePendingClear(const PendingClear& clear, ClearFlushReason reason) {
    if (blit_encoder || render_encoder || compute_encoder) {
        EndEncoding(PassEnd::Clear);
    }
    // The clear writes the textures in this command buffer: side uploads to them can't be
    // ordered before it anymore.
    for (size_t i = 0; i < clear.num_textures; ++i) {
        MarkTextureUsed(clear.textures[i]);
    }
    @autoreleasepool {
        // Nothing draws in this pass: the aspects it clears are stored, the attachments it
        // doesn't touch keep their memory contents without a store. The descriptor is a copy of
        // the framebuffer key owned by this clear, so the store actions the key leaves unknown
        // are set here.
        for (NSUInteger index = 0; index < MAX_PASS_TEXTURES - 2; ++index) {
            MTLRenderPassColorAttachmentDescriptor* color =
                clear.descriptor.colorAttachments[index];
            if (color.texture != nil && color.loadAction != MTLLoadActionClear) {
                color.storeAction = MTLStoreActionDontCare;
            }
        }
        MTLRenderPassDepthAttachmentDescriptor* depth = clear.descriptor.depthAttachment;
        if (depth.texture != nil) {
            if (depth.loadAction == MTLLoadActionClear) {
                depth.storeAction = MTLStoreActionStore;
            } else {
                depth.storeAction = MTLStoreActionDontCare;
                ++stats.depth_dontcare_stores;
            }
        }
        MTLRenderPassStencilAttachmentDescriptor* stencil = clear.descriptor.stencilAttachment;
        if (stencil.texture != nil) {
            if (stencil.loadAction == MTLLoadActionClear) {
                stencil.storeAction = MTLStoreActionStore;
            } else {
                stencil.storeAction = MTLStoreActionDontCare;
                ++stats.stencil_dontcare_stores;
            }
        }
        id<MTLRenderCommandEncoder> encoder =
            [CommandBuffer() renderCommandEncoderWithDescriptor:clear.descriptor];
        if (!encoder) {
            LOG_ERROR(Render_Metal, "Failed to create the render command encoder of a clear");
            return;
        }
        [encoder endEncoding];
    }
    ++stats.render_passes;
    ++stats.flushed_clears;
    switch (reason) {
    case ClearFlushReason::Evicted:
        ++stats.clears_run_evicted;
        break;
    case ClearFlushReason::ImageOperation:
        ++stats.clears_run_image_op;
        break;
    case ClearFlushReason::BoundView:
        ++stats.clears_run_bound_view;
        break;
    case ClearFlushReason::PassOverlap:
        ++stats.clears_run_overlap;
        break;
    case ClearFlushReason::Flush:
        ++stats.clears_run_flush;
        break;
    }
}

size_t Scheduler::PassTextures(MTLRenderPassDescriptor* descriptor,
                               std::array<id<MTLTexture>, MAX_PASS_TEXTURES>& textures) {
    size_t count = 0;
    const auto add = [&](id<MTLTexture> texture) {
        if (!texture || count >= textures.size()) {
            return;
        }
        const id<MTLTexture> root = RootTexture(texture);
        for (size_t i = 0; i < count; ++i) {
            if (textures[i] == root) {
                return;
            }
        }
        textures[count++] = root;
    };
    @autoreleasepool {
        for (NSUInteger index = 0; index < MAX_PASS_TEXTURES - 2; ++index) {
            add(descriptor.colorAttachments[index].texture);
        }
        add(descriptor.depthAttachment.texture);
        add(descriptor.stencilAttachment.texture);
    }
    return count;
}

void Scheduler::WaitSubmitted(u64 tick) {
    if (IsFree(tick)) {
        return;
    }
    std::unique_lock lock{tick_mutex};
    tick_cv.wait(lock, [this, tick] { return IsFree(tick); });
}

void Scheduler::OnCompletion(std::function<void()>&& func) {
    pending_callbacks->push_back(std::move(func));
}

void Scheduler::SignalTick(u64 tick) {
    // Notifying with the lock held guarantees that a waiter destroying the scheduler can only
    // run after this function stopped touching it.
    std::scoped_lock lock{tick_mutex};
    // Command buffers of a queue complete in submission order, max() only guards against a
    // driver reporting them out of order.
    if (gpu_tick.load(std::memory_order_relaxed) < tick) {
        gpu_tick.store(tick, std::memory_order_release);
    }
    tick_cv.notify_all();
}

StagingBufferPool::StagingBufferPool(const Device& device_, Scheduler& scheduler_)
    : device{device_.GetMTLDevice()}, scheduler{scheduler_} {}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    const u64 request_size = (std::max)(static_cast<u64>(size), u64{1});
    const u32 log2 = (std::max)(MIN_LOG2, static_cast<u32>(Common::Log2Ceil(request_size)));
    if (log2 >= NUM_LEVELS) {
        LOG_CRITICAL(Render_Metal, "Staging request of {} bytes is too large", size);
        throw std::bad_alloc{};
    }
    std::optional<StagingBufferRef> ref = TryReuse(usage, log2, deferred);
    if (!ref) {
        ref = Create(usage, log2, deferred);
    }
    ref->mapped_span = ref->mapped_span.first(size);
    return *ref;
}

std::optional<StagingBufferRef> StagingBufferPool::TryReuse(MemoryUsage usage, u32 log2,
                                                            bool deferred) {
    Level& level = CacheFor(usage)[log2];
    const size_t num_entries = level.entries.size();
    for (size_t step = 0; step < num_entries; ++step) {
        const size_t index = (level.iterate_index + step) % num_entries;
        Entry& entry = level.entries[index];
        if (!entry.buffer || entry.deferred || !scheduler.IsFree(entry.tick)) {
            continue;
        }
        level.iterate_index = index + 1;
        entry.tick = scheduler.CurrentTick();
        entry.frame = frame;
        entry.deferred = deferred;
        return StagingBufferRef{
            .buffer = entry.buffer,
            .offset = 0,
            .mapped_span = entry.mapped_span,
            .usage = usage,
            .log2_level = log2,
            .index = index,
            .deferred = deferred,
        };
    }
    return std::nullopt;
}

StagingBufferRef StagingBufferPool::Create(MemoryUsage usage, u32 log2, bool deferred) {
    const NSUInteger size = NSUInteger{1} << log2;
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
    if (!buffer) {
        LOG_CRITICAL(Render_Metal, "Failed to allocate {} bytes of staging memory", size);
        throw std::bad_alloc{};
    }
    buffer.label = usage == MemoryUsage::Upload ? @"Eden upload staging" : @"Eden download staging";

    Level& level = CacheFor(usage)[log2];
    const auto free_slot = std::ranges::find_if(level.entries,
                                                [](const Entry& entry) { return !entry.buffer; });
    const size_t index = static_cast<size_t>(free_slot - level.entries.begin());
    if (free_slot == level.entries.end()) {
        level.entries.emplace_back();
    }
    Entry& entry = level.entries[index];
    entry.buffer = buffer;
    entry.mapped_span = std::span(static_cast<u8*>(buffer.contents), size);
    allocated_bytes += size;
    entry.tick = scheduler.CurrentTick();
    entry.frame = frame;
    entry.deferred = deferred;
    return StagingBufferRef{
        .buffer = buffer,
        .offset = 0,
        .mapped_span = entry.mapped_span,
        .usage = usage,
        .log2_level = log2,
        .index = index,
        .deferred = deferred,
    };
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    if (!ref.buffer || !ref.deferred) {
        return;
    }
    Level& level = CacheFor(ref.usage)[ref.log2_level];
    ASSERT(ref.index < level.entries.size());
    Entry& entry = level.entries[ref.index];
    ASSERT(entry.buffer == ref.buffer && entry.deferred);
    // The GPU may still be using it through commands recorded in this tick.
    entry.tick = scheduler.CurrentTick();
    entry.frame = frame;
    entry.deferred = false;
    ref.deferred = false;
}

StagingBufferRef StagingBufferPool::RequestStream(size_t size, size_t alignment) {
    return RequestFromStream(stream, size, alignment);
}

StagingBufferRef StagingBufferPool::RequestSideStream(size_t size) {
    return RequestFromStream(side_stream, size, BLIT_ALIGNMENT);
}

StagingBufferRef StagingBufferPool::RequestFromStream(Stream& stream_, size_t size,
                                                      size_t alignment) {
    if (size > STREAM_CHUNK_SIZE) {
        return Request(size, MemoryUsage::Upload);
    }
    stream_.offset = Common::AlignUp(stream_.offset, alignment);
    if (!stream_.chunk || stream_.offset + size > STREAM_CHUNK_SIZE) {
        if (stream_.chunk) {
            FreeDeferred(*stream_.chunk);
        }
        // Held as deferred so that the pool never recycles the chunk while it is current.
        stream_.chunk = Request(STREAM_CHUNK_SIZE, MemoryUsage::Upload, true);
        stream_.offset = 0;
    }
    Entry& entry = upload_cache[stream_.chunk->log2_level].entries[stream_.chunk->index];
    entry.tick = scheduler.CurrentTick();
    entry.frame = frame;

    StagingBufferRef ref = *stream_.chunk;
    ref.offset = stream_.offset;
    ref.mapped_span = stream_.chunk->mapped_span.subspan(stream_.offset, size);
    ref.deferred = false;
    stream_.offset += size;
    return ref;
}

void StagingBufferPool::TickFrame() {
    ++frame;
    ReleaseUnused(upload_cache);
    ReleaseUnused(download_cache);
}

void StagingBufferPool::ReleaseUnused(Cache& cache) {
    // Every level is scanned each frame, they hold a few entries. Large buffers are released
    // sooner: at low frame rates hundreds of frames are minutes of wasted unified memory.
    static constexpr u64 FRAMES_TO_KEEP = 120;
    static constexpr u64 FRAMES_TO_KEEP_LARGE = 12;
    static constexpr u32 LARGE_LOG2 = 22;
    for (u32 log2 = 0; log2 < NUM_LEVELS; ++log2) {
        const u64 frames_to_keep = log2 >= LARGE_LOG2 ? FRAMES_TO_KEEP_LARGE : FRAMES_TO_KEEP;
        for (Entry& entry : cache[log2].entries) {
            if (entry.buffer && !entry.deferred && scheduler.IsFree(entry.tick) &&
                frame - entry.frame > frames_to_keep) {
                allocated_bytes -= entry.mapped_span.size();
                entry.buffer = nil;
                entry.mapped_span = {};
            }
        }
    }
}

Buffer::Buffer(BufferCacheRuntime& runtime_, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase(null_params), runtime{&runtime_} {}

Buffer::Buffer(BufferCacheRuntime& runtime_, DAddr cpu_addr_, u64 size_bytes_,
               [[maybe_unused]] bool sparse_compatible)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), runtime{&runtime_} {
    // Shared storage lets the CPU write guest data straight into the allocation on unified
    // memory, the GPU reads it without any intermediate copy.
    buffer = [runtime->device.GetMTLDevice() newBufferWithLength:SizeBytes()
                                                         options:MTLResourceStorageModeShared];
    if (!buffer) {
        LOG_CRITICAL(Render_Metal, "Failed to allocate a buffer of {} bytes", SizeBytes());
        throw std::bad_alloc{};
    }
    mapped = static_cast<u8*>(buffer.contents);
    const u64 granules =
        Common::DivCeil(static_cast<u64>(SizeBytes()), u64{1} << USAGE_GRANULARITY_LOG2);
    current_bits.resize(Common::DivCeil(granules, u64{64}));
    previous_bits.resize(current_bits.size());
    produced_bits.resize(current_bits.size());
    hard_bits.resize(current_bits.size());
}

id<MTLTexture> Buffer::View(u32 offset, u32 size, PixelFormat format) {
    if (!buffer) {
        return nil;
    }
    const auto it = std::ranges::find_if(views, [offset, size, format](const BufferView& view) {
        return offset == view.offset && size == view.size && format == view.format;
    });
    if (it != views.end()) {
        return it->texture;
    }
    const FormatInfo format_info = SurfaceFormat(runtime->device, format);
    const u32 bytes_per_texel = VideoCore::Surface::BytesPerBlock(format);
    if (!format_info.supported || bytes_per_texel == 0 ||
        VideoCore::Surface::DefaultBlockWidth(format) != 1) {
        LOG_WARNING(Render_Metal, "Texture buffer format {} is not supported", format);
        return nil;
    }
    const u32 width = size / bytes_per_texel;
    id<MTLDevice> device = runtime->device.GetMTLDevice();
    const NSUInteger alignment =
        [device minimumTextureBufferAlignmentForPixelFormat:format_info.format];
    if (width == 0 || (alignment != 0 && offset % alignment != 0) ||
        static_cast<u64>(offset) + size > SizeBytes()) {
        LOG_WARNING(Render_Metal, "Texture buffer view at offset {} with {} bytes can't be created",
                    offset, size);
        return nil;
    }
    id<MTLTexture> texture;
    @autoreleasepool {
        MTLTextureUsage usage = MTLTextureUsageShaderRead;
        if (format_info.storage) {
            usage |= MTLTextureUsageShaderWrite;
        }
        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format_info.format
                                                                   width:width
                                                         resourceOptions:buffer.resourceOptions
                                                                   usage:usage];
        const NSUInteger bytes_per_row = static_cast<NSUInteger>(width) * bytes_per_texel;
        texture = [buffer newTextureWithDescriptor:descriptor
                                            offset:offset
                                       bytesPerRow:bytes_per_row];
    }
    // Streamed texture buffers create a view per offset, drop the oldest ones so the linear
    // lookup stays short. Command buffers keep the views they use alive.
    static constexpr size_t MAX_VIEWS = 256;
    if (views.size() >= MAX_VIEWS) {
        views.erase(views.begin(), views.begin() + MAX_VIEWS / 2);
    }
    views.push_back(BufferView{
        .offset = offset,
        .size = size,
        .format = format,
        .texture = texture,
    });
    return texture;
}

namespace {

struct GranuleSpan {
    u64 first_word;
    u64 last_word;
    u64 first_mask;
    u64 last_mask;
};

GranuleSpan MakeGranuleSpan(u64 offset, u64 size, size_t num_words, u64 granularity_log2) {
    const u64 max_granule = num_words * 64 - 1;
    const u64 first = (std::min)(offset >> granularity_log2, max_granule);
    const u64 last = (std::min)((offset + size - 1) >> granularity_log2, max_granule);
    return GranuleSpan{
        .first_word = first / 64,
        .last_word = last / 64,
        .first_mask = ~u64{0} << (first % 64),
        .last_mask = (last % 64 == 63) ? ~u64{0} : ((u64{1} << ((last % 64) + 1)) - 1),
    };
}

bool TestBits(const std::vector<u64>& bits, const GranuleSpan& span) {
    if (span.first_word == span.last_word) {
        return (bits[span.first_word] & (span.first_mask & span.last_mask)) != 0;
    }
    if ((bits[span.first_word] & span.first_mask) != 0) {
        return true;
    }
    for (u64 w = span.first_word + 1; w < span.last_word; ++w) {
        if (bits[w] != 0) {
            return true;
        }
    }
    return (bits[span.last_word] & span.last_mask) != 0;
}

void SetBits(std::vector<u64>& bits, const GranuleSpan& span) {
    if (span.first_word == span.last_word) {
        bits[span.first_word] |= (span.first_mask & span.last_mask);
        return;
    }
    bits[span.first_word] |= span.first_mask;
    for (u64 w = span.first_word + 1; w < span.last_word; ++w) {
        bits[w] = ~u64{0};
    }
    bits[span.last_word] |= span.last_mask;
}

void ClearBits(std::vector<u64>& bits, const GranuleSpan& span) {
    if (span.first_word == span.last_word) {
        bits[span.first_word] &= ~(span.first_mask & span.last_mask);
        return;
    }
    bits[span.first_word] &= ~span.first_mask;
    for (u64 w = span.first_word + 1; w < span.last_word; ++w) {
        bits[w] = 0;
    }
    bits[span.last_word] &= ~span.last_mask;
}

} // Anonymous namespace

id<MTLTexture> Buffer::ResolveView(u32 offset, u32 size, PixelFormat format, bool is_written) {
    PruneOverlays();
    if (!overlays.empty()) {
        UploadOverlay* overlay = OverlayCovering(offset, size);
        if (!overlay) {
            overlay = ExtendOverlay(offset, size);
        }
        if (overlay) {
            // The view aliases the staging copy, like a binding resolved there. Texture buffer
            // views need the format's alignment, which the copy's offset may not satisfy.
            const FormatInfo format_info = SurfaceFormat(runtime->device, format);
            const u32 bytes_per_texel = VideoCore::Surface::BytesPerBlock(format);
            const u64 view_offset = overlay->staging_offset + (offset - overlay->offset);
            id<MTLDevice> device = runtime->device.GetMTLDevice();
            const NSUInteger alignment =
                format_info.supported
                    ? [device minimumTextureBufferAlignmentForPixelFormat:format_info.format]
                    : 0;
            const u32 width = bytes_per_texel != 0 ? size / bytes_per_texel : 0;
            if (format_info.supported && width != 0 &&
                VideoCore::Surface::DefaultBlockWidth(format) == 1 &&
                (alignment == 0 || view_offset % alignment == 0)) {
                if (is_written) {
                    overlay->gpu_write_tick = runtime->scheduler.CurrentTick();
                }
                id<MTLTexture> texture;
                @autoreleasepool {
                    MTLTextureUsage usage = MTLTextureUsageShaderRead;
                    if (format_info.storage) {
                        usage |= MTLTextureUsageShaderWrite;
                    }
                    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                        textureBufferDescriptorWithPixelFormat:format_info.format
                                                         width:width
                                               resourceOptions:overlay->staging.resourceOptions
                                                         usage:usage];
                    texture = [overlay->staging
                        newTextureWithDescriptor:descriptor
                                          offset:view_offset
                                     bytesPerRow:static_cast<NSUInteger>(width) * bytes_per_texel];
                }
                if (texture) {
                    return texture;
                }
            }
        }
        // Texture views of the buffer itself need the deferred uploads in it.
        ApplyOverlaysOverlapping(offset, size);
    }
    return View(offset, size, format);
}

void Buffer::ImmediateUpload(size_t offset, std::span<const u8> data) {
    if (!buffer || data.empty()) {
        return;
    }
    ASSERT(offset + data.size() <= SizeBytes());
    const u64 size = data.size();
    const u64 end = offset + size;
    PruneOverlays();
    if (!overlays.empty()) {
        const auto first = FirstOverlayEndingAfter(offset);
        auto last = first;
        while (last != overlays.end() && last->offset < end) {
            ++last;
        }
        if (first != last) {
            const bool single = std::next(first) == last;
            if (single && first->offset <= offset && end <= first->offset + first->size &&
                first->gpu_write_tick != 0) {
                // Shaders write the staging copy, its bytes can't be patched on the CPU.
                runtime->ApplyPendingUploads();
                overlays.clear();
                runtime->scheduler.CountOverlayApply();
            } else if (single && first->offset <= offset && end <= first->offset + first->size) {
                // The range is inside a deferred upload of this pass: the bytes patch a copy of
                // its staging data, which later draws read, while draws already recorded keep
                // the copy they were bound to. The copy at the end of the pass is the patched
                // range, encoded after the one of the earlier upload.
                UploadOverlay& overlay = *first;
                const u64 patch_offset = offset - overlay.offset;
                const StagingBufferRef staging =
                    runtime->staging_pool.Request(overlay.size, MemoryUsage::Upload);
                if (overlay.size != size) {
                    std::memcpy(staging.mapped_span.data(),
                                static_cast<const u8*>(overlay.staging.contents) +
                                    overlay.staging_offset,
                                overlay.size);
                }
                std::memcpy(staging.mapped_span.data() + patch_offset, data.data(), size);
                overlay.staging = staging.buffer;
                overlay.staging_offset = staging.offset;
                ReplacePendingUpload(offset, size, staging.buffer, staging.offset + patch_offset);
                runtime->scheduler.CountDeferredUpload();
                runtime->scheduler.CountOverlayMerge();
                return;
            } else {
                const bool contained =
                    std::all_of(first, last, [&](const UploadOverlay& overlay) {
                        return offset <= overlay.offset && overlay.offset + overlay.size <= end;
                    });
                if (contained) {
                    // The new data supersedes them; their copies still run first at the end
                    // of the encoder and the copy of this upload lands last.
                    overlays.erase(first, last);
                    runtime->scheduler.CountOverlayMerge();
                } else if (ExtendOverlay(offset, size)) {
                    // Straddling deferred uploads: grown into one that covers this range, the
                    // bytes then patch its staging copy like any upload inside an overlay.
                    ImmediateUpload(offset, data);
                    return;
                } else {
                    // Every pending copy is applied now, which ends the open encoder.
                    runtime->ApplyPendingUploads();
                    overlays.clear();
                    runtime->scheduler.CountOverlayApply();
                }
            }
        }
    }
    // Whatever the GPU produced here, the CPU overwrites it: guest memory is current again.
    ClearProducedBits(offset, size);
    PruneSideUploads();
    if (!side_uploads.empty() && !IsRegionUsedByCurrentTick(offset, size)) {
        if (SideUpload* const side = SideUploadCovering(offset, size)) {
            // The side command buffer reads the staging copy when it runs, after this write.
            std::memcpy(static_cast<u8*>(side->staging.contents) + side->staging_offset +
                            (offset - side->offset),
                        data.data(), size);
            runtime->scheduler.CountSidePatch();
            return;
        }
        if (HasSideUploadOverlapping(offset, size) && IsBlitAligned(0, offset, size)) {
            // Straddling side uploads of this tick: a later side copy wins over them.
            SideUploadBytes(offset, data);
            return;
        }
    }
    if (!IsRegionUsed(offset, size)) {
        std::memcpy(mapped + offset, data.data(), size);
        return;
    }
    if (!IsRegionUsedByCurrentTick(offset, size) && IsBlitAligned(0, offset, size) &&
        side_uploads.size() < MAX_SIDE_UPLOADS) {
        // Only earlier submissions use the region: the side command buffer copies the data
        // after them and before anything of this tick, so bindings keep using the buffer.
        SideUploadBytes(offset, data);
        return;
    }
    // Commands already recorded still need the old contents, so the write goes through staging
    // memory and is ordered after them on the GPU.
    const StagingBufferRef staging = runtime->staging_pool.Request(size, MemoryUsage::Upload);
    std::memcpy(staging.mapped_span.data(), data.data(), size);
    MarkGpuWrite(offset, size, GpuWriteKind::Upload);
    Scheduler::UploadBlitReason reason = Scheduler::UploadBlitReason::Unaligned;
    if (!IsBlitAligned(staging.offset, offset, size)) {
        reason = Scheduler::UploadBlitReason::Unaligned;
    } else if (overlays.size() >= MAX_OVERLAYS) {
        reason = Scheduler::UploadBlitReason::TooManyOverlays;
    } else {
        // Copying now would end the open encoder: a render pass, which tile based GPUs pay
        // with a store and a reload of every attachment, or a compute encoder shared by
        // consecutive dispatches. The copy waits for the end of the encoder instead and the
        // draws and dispatches until then are bound to the staging copy.
        const auto position = std::ranges::upper_bound(overlays, offset, {}, &UploadOverlay::offset);
        overlays.insert(position, UploadOverlay{
                                      .offset = offset,
                                      .size = size,
                                      .generation = runtime->overlay_generation,
                                      .staging = staging.buffer,
                                      .staging_offset = staging.offset,
                                  });
        runtime->pending_uploads.push_back(BufferCacheRuntime::PendingUpload{
            .destination = buffer,
            .destination_offset = offset,
            .staging = staging.buffer,
            .staging_offset = staging.offset,
            .size = size,
        });
        runtime->scheduler.CountDeferredUpload();
        return;
    }
    runtime->scheduler.CountBufferUploadBlit(reason);
    runtime->EncodeCopy(staging.buffer, staging.offset, buffer, offset, size);
}

void Buffer::ReplacePendingUpload(u64 offset, u64 size, id<MTLBuffer> staging,
                                  u64 staging_offset) {
    // The latest copy of a range is the one that matters, the earlier one of the same range
    // would only be overwritten. Recent uploads sit at the back.
    auto& uploads = runtime->pending_uploads;
    for (auto it = uploads.rbegin(); it != uploads.rend(); ++it) {
        if (it->destination == buffer && it->destination_offset == offset && it->size == size) {
            it->staging = staging;
            it->staging_offset = staging_offset;
            return;
        }
    }
    uploads.push_back(BufferCacheRuntime::PendingUpload{
        .destination = buffer,
        .destination_offset = offset,
        .staging = staging,
        .staging_offset = staging_offset,
        .size = size,
    });
}

void Buffer::PruneOverlays() {
    if (overlays.empty()) {
        return;
    }
    const u64 generation = runtime->OverlayGeneration();
    std::erase_if(overlays, [generation](const UploadOverlay& overlay) {
        return overlay.generation != generation;
    });
}

std::vector<Buffer::UploadOverlay>::iterator Buffer::FirstOverlayEndingAfter(u64 offset) noexcept {
    // Disjoint and sorted by offset, so the ends are sorted as well.
    return std::ranges::upper_bound(overlays, offset, {}, [](const UploadOverlay& overlay) {
        return overlay.offset + overlay.size;
    });
}

Buffer::UploadOverlay* Buffer::OverlayCovering(u64 offset, u64 size) noexcept {
    const auto it = FirstOverlayEndingAfter(offset);
    if (it == overlays.end() || it->offset > offset || offset + size > it->offset + it->size) {
        return nullptr;
    }
    return &*it;
}

BufferBinding Buffer::Resolve(u64 offset, u64 size, bool is_written) {
    PruneOverlays();
    if (!overlays.empty()) {
        UploadOverlay* overlay = OverlayCovering(offset, size);
        if (!overlay) {
            // Bindings are usually larger than the bytes the guest rewrote (a whole constant
            // buffer against one updated slot): the deferred upload grows to the binding.
            overlay = ExtendOverlay(offset, size);
        }
        if (overlay) {
            if (is_written) {
                // The shaders write the staging copy; the copy into the buffer at the end of
                // the encoder carries their output, CPU reads of the copy wait for them.
                overlay->gpu_write_tick = runtime->scheduler.CurrentTick();
            }
            return BufferBinding{
                .buffer = overlay->staging,
                .offset = overlay->staging_offset + (offset - overlay->offset),
                .size = size,
            };
        }
        ApplyOverlaysOverlapping(offset, size);
    }
    return BufferBinding{.buffer = buffer, .offset = offset, .size = size};
}

Buffer::UploadOverlay* Buffer::ExtendOverlay(u64 offset, u64 size) {
    // Maxwell constant buffers are at most 64 KiB, larger ranges are storage buffers whose
    // copies would cost more than the pass they save.
    static constexpr u64 MAX_EXTENSION = 64 * 1024;
    const u64 end = offset + size;
    const auto first = FirstOverlayEndingAfter(offset);
    auto last = first;
    u64 begin = offset;
    u64 extended_end = end;
    for (; last != overlays.end() && last->offset < end; ++last) {
        if (last->gpu_write_tick != 0) {
            // Shaders write the staging copy, its bytes can't be read on the CPU yet.
            return nullptr;
        }
        begin = (std::min)(begin, last->offset);
        extended_end = (std::max)(extended_end, last->offset + last->size);
    }
    if (first == last) {
        return nullptr;
    }
    // Bindings start anywhere, the copy has to be 4 byte aligned.
    begin = Common::AlignDown(begin, BLIT_ALIGNMENT);
    extended_end = Common::AlignUp(extended_end, BLIT_ALIGNMENT);
    const u64 total = extended_end - begin;
    if (total > MAX_EXTENSION || extended_end > SizeBytes()) {
        return nullptr;
    }
    // The bytes between the uploads are the latest the CPU can see, then the uploads go on top.
    const StagingBufferRef staging = runtime->staging_pool.Request(total, MemoryUsage::Upload);
    u8* const bytes = staging.mapped_span.data();
    if (!ReadLatestBytes(begin, total, bytes)) {
        return nullptr;
    }
    for (auto it = first; it != last; ++it) {
        std::memcpy(bytes + (it->offset - begin),
                    static_cast<const u8*>(it->staging.contents) + it->staging_offset, it->size);
    }
    // The merged copy is encoded after the ones it absorbs; outside them it rewrites bytes
    // the buffer already holds.
    runtime->pending_uploads.push_back(BufferCacheRuntime::PendingUpload{
        .destination = buffer,
        .destination_offset = begin,
        .staging = staging.buffer,
        .staging_offset = staging.offset,
        .size = total,
    });
    MarkGpuWrite(begin, total, GpuWriteKind::Upload);
    const auto position = overlays.erase(first, last);
    const auto inserted = overlays.insert(position, UploadOverlay{
                                                        .offset = begin,
                                                        .size = total,
                                                        .generation = runtime->overlay_generation,
                                                        .staging = staging.buffer,
                                                        .staging_offset = staging.offset,
                                                    });
    runtime->scheduler.CountOverlayExtension();
    return &*inserted;
}

void Buffer::ApplyOverlaysOverlapping(u64 offset, u64 size) {
    PruneOverlays();
    if (overlays.empty()) {
        return;
    }
    const auto it = FirstOverlayEndingAfter(offset);
    if (it != overlays.end() && it->offset < offset + size) {
        // Rare: every deferred upload is applied, which keeps their order simple.
        runtime->ApplyPendingUploads();
        overlays.clear();
        runtime->scheduler.CountOverlayApply();
    }
}

const u8* Buffer::ReadableBytes(u64 offset, u64 size, u64& wait_tick) {
    PruneOverlays();
    const UploadOverlay* overlay = OverlayCovering(offset, size);
    if (!overlay && !overlays.empty()) {
        overlay = ExtendOverlay(offset, size);
    }
    if (overlay) {
        wait_tick = overlay->gpu_write_tick;
        return static_cast<const u8*>(overlay->staging.contents) + overlay->staging_offset +
               (offset - overlay->offset);
    }
    PruneSideUploads();
    if (const SideUpload* const side = SideUploadCovering(offset, size)) {
        // Written by the CPU only, the copy into the buffer is still queued.
        wait_tick = 0;
        return static_cast<const u8*>(side->staging.contents) + side->staging_offset +
               (offset - side->offset);
    }
    ApplyOverlaysOverlapping(offset, size);
    wait_tick = PendingWriteTick(offset, size);
    return mapped + offset;
}

void Buffer::PruneSideUploads() {
    if (side_uploads.empty()) {
        return;
    }
    const u64 tick = runtime->scheduler.CurrentTick();
    std::erase_if(side_uploads, [tick](const SideUpload& side) { return side.tick != tick; });
}

Buffer::SideUpload* Buffer::SideUploadCovering(u64 offset, u64 size) noexcept {
    for (auto it = side_uploads.rbegin(); it != side_uploads.rend(); ++it) {
        if (it->offset <= offset && offset + size <= it->offset + it->size) {
            return &*it;
        }
    }
    return nullptr;
}

bool Buffer::HasSideUploadOverlapping(u64 offset, u64 size) const noexcept {
    return std::ranges::any_of(side_uploads, [offset, size](const SideUpload& side) {
        return side.offset < offset + size && offset < side.offset + side.size;
    });
}

void Buffer::SideUploadBytes(u64 offset, std::span<const u8> data) {
    const u64 size = data.size();
    // Staging comes from the side stream, where consecutive uploads are contiguous: an upload
    // continuing the previous one of this buffer merges into its blit in the scheduler.
    const StagingBufferRef staging = runtime->staging_pool.RequestSideStream(size);
    std::memcpy(staging.mapped_span.data(), data.data(), size);
    runtime->scheduler.SideCopy(staging.buffer, staging.offset, buffer, offset, size);
    // In use until the copy completes, which later rewrites of the range must respect, but
    // not by a recorded command: a rewrite within the tick may still patch the staging copy.
    MarkUsage(offset, size);
    SoftenLastUsage();
    AddPendingWrite(offset, size);
    if (side_uploads.size() >= MAX_SIDE_UPLOADS) {
        side_uploads.erase(side_uploads.begin(), side_uploads.begin() + MAX_SIDE_UPLOADS / 2);
    }
    side_uploads.push_back(SideUpload{
        .offset = offset,
        .size = size,
        .tick = runtime->scheduler.CurrentTick(),
        .staging = staging.buffer,
        .staging_offset = staging.offset,
    });
    runtime->scheduler.CountSideCopy();
}

bool Buffer::ReadLatestBytes(u64 offset, u64 size, u8* bytes) {
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, produced_bits.size(), USAGE_GRANULARITY_LOG2);
    const bool produced = !produced_bits.empty() && TestBits(produced_bits, span);
    if (produced && PendingWriteTick(offset, size) != 0) {
        // GPU produced bytes still being written can't be read from anywhere yet.
        return false;
    }
    // Guest memory holds everything the CPU wrote, including uploads still queued.
    runtime->device_memory.ReadBlockUnsafe(CpuAddr() + offset, bytes, size);
    if (!produced) {
        return true;
    }
    // Bytes the GPU produced only exist in the buffer, its writes completed.
    constexpr u64 granule = u64{1} << USAGE_GRANULARITY_LOG2;
    const u64 end = offset + size;
    for (u64 g = offset >> USAGE_GRANULARITY_LOG2; g <= (end - 1) >> USAGE_GRANULARITY_LOG2; ++g) {
        if (g / 64 >= produced_bits.size() || (produced_bits[g / 64] & (u64{1} << (g % 64))) == 0) {
            continue;
        }
        const u64 begin = (std::max)(g * granule, offset);
        const u64 stop = (std::min)((g + 1) * granule, end);
        std::memcpy(bytes + (begin - offset), mapped + begin, stop - begin);
    }
    return true;
}


void Buffer::RotateUsage(u64 tick) noexcept {
    if (current_usage_tick == tick) {
        return;
    }
    const Scheduler& scheduler = runtime->scheduler;
    if (scheduler.IsFree(current_usage_tick)) {
        // Everything recorded so far completed.
        std::ranges::fill(current_bits, u64{0});
        std::ranges::fill(previous_bits, u64{0});
        previous_usage_tick = 0;
    } else if (scheduler.IsFree(previous_usage_tick)) {
        current_bits.swap(previous_bits);
        std::ranges::fill(current_bits, u64{0});
        previous_usage_tick = current_usage_tick;
    } else {
        for (size_t i = 0; i < current_bits.size(); ++i) {
            previous_bits[i] |= current_bits[i];
        }
        std::ranges::fill(current_bits, u64{0});
        previous_usage_tick = current_usage_tick;
    }
    std::ranges::fill(hard_bits, u64{0});
    current_usage_tick = tick;
}

void Buffer::MarkUsage(u64 offset, u64 size) noexcept {
    if (current_bits.empty() || size == 0) {
        return;
    }
    const u64 tick = runtime->scheduler.CurrentTick();
    RotateUsage(tick);
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, current_bits.size(), USAGE_GRANULARITY_LOG2);
    SetBits(current_bits, span);
    // Hard uses of the tick, remembering what this call added so the caller can soften it.
    last_mark.clear();
    for (u64 w = span.first_word; w <= span.last_word; ++w) {
        u64 mask = ~u64{0};
        if (w == span.first_word) {
            mask &= span.first_mask;
        }
        if (w == span.last_word) {
            mask &= span.last_mask;
        }
        const u64 added = mask & ~hard_bits[w];
        if (added != 0) {
            hard_bits[w] |= added;
            last_mark.emplace_back(static_cast<u32>(w), added);
        }
    }
    last_usage_tick = tick;
}

void Buffer::SoftenLastUsage() noexcept {
    for (const auto& [word, bits] : last_mark) {
        hard_bits[word] &= ~bits;
    }
    last_mark.clear();
}

void Buffer::MarkGpuWrite(u64 offset, u64 size, GpuWriteKind kind) noexcept {
    if (size == 0) {
        return;
    }
    // A pending write is also a use: CPU writes to the region must be ordered after it.
    MarkUsage(offset, size);
    AddPendingWrite(offset, size);
    if (kind == GpuWriteKind::Produced && !produced_bits.empty()) {
        const GranuleSpan span =
            MakeGranuleSpan(offset, size, produced_bits.size(), USAGE_GRANULARITY_LOG2);
        SetBits(produced_bits, span);
    }
}

void Buffer::ClearProducedBits(u64 offset, u64 size) noexcept {
    if (produced_bits.empty() || size == 0) {
        return;
    }
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, produced_bits.size(), USAGE_GRANULARITY_LOG2);
    ClearBits(produced_bits, span);
}

bool Buffer::HasGpuProducedData(u64 offset, u64 size) const noexcept {
    if (produced_bits.empty() || size == 0) {
        return false;
    }
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, produced_bits.size(), USAGE_GRANULARITY_LOG2);
    return TestBits(produced_bits, span);
}

bool Buffer::IsRegionUsedByCurrentTick(u64 offset, u64 size) const noexcept {
    if (current_bits.empty() || size == 0) {
        return false;
    }
    const Scheduler& scheduler = runtime->scheduler;
    if (current_usage_tick != scheduler.CurrentTick() || scheduler.IsFree(current_usage_tick)) {
        return false;
    }
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, current_bits.size(), USAGE_GRANULARITY_LOG2);
    return TestBits(hard_bits, span);
}

void Buffer::AddPendingWrite(u64 offset, u64 size) noexcept {
    if (size == 0) {
        return;
    }
    const Scheduler& scheduler = runtime->scheduler;
    const u64 tick = scheduler.CurrentTick();
    const u64 end = offset + size;
    std::erase_if(pending_writes,
                  [&scheduler](const WriteRange& range) { return scheduler.IsFree(range.tick); });
    // Ranges of the same tick are merged when they touch, the common case of repeated binds.
    for (WriteRange& range : pending_writes) {
        if (range.tick == tick && offset <= range.end && range.begin <= end) {
            range.begin = (std::min)(range.begin, offset);
            range.end = (std::max)(range.end, end);
            return;
        }
    }
    if (pending_writes.size() >= MAX_WRITE_RANGES) {
        WriteRange merged{offset, end, tick};
        for (const WriteRange& range : pending_writes) {
            merged.begin = (std::min)(merged.begin, range.begin);
            merged.end = (std::max)(merged.end, range.end);
        }
        pending_writes.clear();
        pending_writes.push_back(merged);
    } else {
        pending_writes.push_back(WriteRange{offset, end, tick});
    }
}

bool Buffer::IsRegionUsed(u64 offset, u64 size) const noexcept {
    if (current_bits.empty() || size == 0) {
        return false;
    }
    const Scheduler& scheduler = runtime->scheduler;
    const bool current_pending = !scheduler.IsFree(current_usage_tick);
    const bool previous_pending = previous_usage_tick != 0 && !scheduler.IsFree(previous_usage_tick);
    if (!current_pending && !previous_pending) {
        return false;
    }
    const GranuleSpan span =
        MakeGranuleSpan(offset, size, current_bits.size(), USAGE_GRANULARITY_LOG2);
    return (current_pending && TestBits(current_bits, span)) ||
           (previous_pending && TestBits(previous_bits, span));
}

u64 Buffer::PendingWriteTick(u64 offset, u64 size) const noexcept {
    const Scheduler& scheduler = runtime->scheduler;
    const u64 end = offset + size;
    u64 tick = 0;
    for (const WriteRange& range : pending_writes) {
        if (offset < range.end && range.begin < end && !scheduler.IsFree(range.tick)) {
            tick = (std::max)(tick, range.tick);
        }
    }
    return tick;
}

void Buffer::ResetUsageTracking() noexcept {
    std::ranges::fill(current_bits, u64{0});
    std::ranges::fill(previous_bits, u64{0});
    std::ranges::fill(hard_bits, u64{0});
    last_mark.clear();
    current_usage_tick = 0;
    previous_usage_tick = 0;
    std::erase_if(pending_writes, [this](const WriteRange& range) {
        return runtime->scheduler.IsFree(range.tick);
    });
}

BufferCacheRuntime::BufferCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                       StagingBufferPool& staging_pool_,
                                       Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device{device_}, scheduler{scheduler_}, staging_pool{staging_pool_},
      device_memory{device_memory_} {
    scheduler.SetEncoderEndCallback([this] { ApplyPendingUploads(); });
}

BufferCacheRuntime::~BufferCacheRuntime() {
    scheduler.SetEncoderEndCallback({});
}

void BufferCacheRuntime::ApplyPendingUploads() {
    if (pending_uploads.empty()) {
        return;
    }
    // Taken first: encoding the copies may end a render pass, which calls back here.
    ++overlay_generation;
    std::vector<PendingUpload> uploads = std::move(pending_uploads);
    pending_uploads.clear();
    for (const PendingUpload& upload : uploads) {
        EncodeCopy(upload.staging, upload.staging_offset, upload.destination,
                   upload.destination_offset, upload.size);
    }
}

void BufferCacheRuntime::TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept {
    for (auto it = slot_buffers.begin(); it != slot_buffers.end(); ++it) {
        if (scheduler.IsFree(it->LastUsageTick())) {
            it->ResetUsageTracking();
        }
    }
    // The staging pool is shared with the texture cache, it is ticked once per frame here.
    staging_pool.TickFrame();
}

void BufferCacheRuntime::ReportMemoryUsage() const {
    // Unified memory pressure makes macOS swap and stalls every thread, the rasterizer logs
    // this with its frame statistics.
    constexpr f64 MiB = 1024.0 * 1024.0;
    const f64 allocated = static_cast<f64>(device.GetMTLDevice().currentAllocatedSize) / MiB;
    const f64 staging = static_cast<f64>(staging_pool.AllocatedBytes()) / MiB;
    const f64 working_set = static_cast<f64>(device.GetRecommendedMaxWorkingSetSize()) / MiB;
    LOG_INFO(Render_Metal,
             "Metal memory: {:.0f} MiB allocated (staging {:.0f} MiB), recommended working set "
             "{:.0f} MiB",
             allocated, staging, working_set);
}

u64 BufferCacheRuntime::CurrentTick() const noexcept {
    return scheduler.CurrentTick();
}

bool BufferCacheRuntime::IsFree(u64 tick) const noexcept {
    return scheduler.IsFree(tick);
}

void BufferCacheRuntime::Wait(u64 tick) {
    scheduler.Wait(tick);
}

void BufferCacheRuntime::Finish() {
    // The buffer cache calls this after a synchronous download, which already waited for the
    // GPU writes of the bytes it read and copied them out of the shared allocation. Nothing
    // else needs the queue drained here.
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    // The caches account for their own allocations; the working set only sizes the garbage
    // collector thresholds, which must stay well above a game's working set.
    return device.GetRecommendedMaxWorkingSetSize();
}

u64 BufferCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetMTLDevice().currentAllocatedSize;
}

u32 BufferCacheRuntime::GetUniformBufferAlignment() const noexcept {
    // Apple GPUs bind constant buffers at any 4 byte aligned offset, other GPUs need 256.
    return device.GetFeatures().apple_gpu_family != 0 ? 4 : 256;
}

u32 BufferCacheRuntime::GetStorageBufferAlignment() const noexcept {
    return 16;
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, MemoryUsage::Upload);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Download, deferred);
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

// Uploads either write the shared allocation directly or record blits, which already end the
// open render pass. Ending only the blit encoder orders the copies against later passes without
// splitting a render pass on every uniform upload, which is expensive on tile based GPUs.
void BufferCacheRuntime::PreCopyBarrier() {
    scheduler.EndBlitEncoding();
}

void BufferCacheRuntime::PostCopyBarrier() {
    scheduler.EndBlitEncoding();
}

void BufferCacheRuntime::EncodeCopy(id<MTLBuffer> src, u64 src_offset, id<MTLBuffer> dst,
                                    u64 dst_offset, u64 size) {
    if (!src || !dst || size == 0) {
        return;
    }
    if (IsBlitAligned(src_offset, dst_offset, size)) {
        [scheduler.BlitEncoder() copyFromBuffer:src
                                   sourceOffset:src_offset
                                       toBuffer:dst
                              destinationOffset:dst_offset
                                           size:size];
        return;
    }
    // TODO: Use a compute kernel for unaligned copies instead of waiting for the GPU.
    static bool logged = false;
    if (!logged) {
        LOG_WARNING(Render_Metal, "Unaligned buffer copy, synchronizing with the GPU");
        logged = true;
    }
    scheduler.Flush();
    scheduler.Wait(scheduler.CurrentTick() - 1, Scheduler::WaitReason::UnalignedCopy);
    std::memmove(static_cast<u8*>(dst.contents) + dst_offset,
                 static_cast<const u8*>(src.contents) + src_offset, size);
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst, Buffer& src,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier) {
    if (!dst.Handle() || !src.Handle()) {
        return;
    }
    for (const VideoCommon::BufferCopy& copy : copies) {
        src.ApplyOverlaysOverlapping(copy.src_offset, copy.size);
        dst.ApplyOverlaysOverlapping(copy.dst_offset, copy.size);
    }
    // Both allocations are shared: when the GPU neither writes the source nor uses the
    // destination, the copy is done on the CPU without touching the open render pass.
    const bool cpu_copy = std::ranges::all_of(copies, [&](const VideoCommon::BufferCopy& copy) {
        return src.PendingWriteTick(copy.src_offset, copy.size) == 0 &&
               !dst.IsRegionUsed(copy.dst_offset, copy.size);
    });
    if (cpu_copy) {
        for (const VideoCommon::BufferCopy& copy : copies) {
            std::memmove(dst.Mapped() + copy.dst_offset, src.Mapped() + copy.src_offset,
                         copy.size);
        }
        return;
    }
    if (barrier) {
        scheduler.EndEncoding(Scheduler::PassEnd::BufferBlit);
    }
    for (const VideoCommon::BufferCopy& copy : copies) {
        src.MarkUsage(copy.src_offset, copy.size);
        dst.MarkUsage(copy.dst_offset, copy.size);
        dst.MarkGpuWrite(copy.dst_offset, copy.size, Buffer::GpuWriteKind::Produced);
        EncodeCopy(src.Handle(), copy.src_offset, dst.Handle(), copy.dst_offset, copy.size);
    }
    if (barrier) {
        scheduler.EndEncoding(Scheduler::PassEnd::BufferBlit);
    }
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst, id<MTLBuffer> src,
                                    std::span<const VideoCommon::BufferCopy> copies,
                                    [[maybe_unused]] bool barrier,
                                    [[maybe_unused]] bool can_reorder_upload) {
    if (!dst.Handle() || !src) {
        return;
    }
    // Only CPU written upload staging (inline memory writes) reaches this overload: the bytes
    // take the same path as direct uploads, deferred when the region is in use by the GPU.
    const u8* const src_pointer = static_cast<const u8*>(src.contents);
    for (const VideoCommon::BufferCopy& copy : copies) {
        dst.ImmediateUpload(copy.dst_offset,
                            std::span<const u8>(src_pointer + copy.src_offset, copy.size));
    }
}

void BufferCacheRuntime::CopyBuffer(id<MTLBuffer> dst, Buffer& src,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier) {
    if (!dst || !src.Handle()) {
        return;
    }
    u64 wait_tick = 0;
    for (const VideoCommon::BufferCopy& copy : copies) {
        src.ApplyOverlaysOverlapping(copy.src_offset, copy.size);
        wait_tick = (std::max)(wait_tick, src.PendingWriteTick(copy.src_offset, copy.size));
    }
    if (wait_tick != 0 && !barrier) {
        // Asynchronous downloads are read after a fence, order a GPU copy instead of waiting.
        scheduler.CountBufferDownloadCopy();
        for (const VideoCommon::BufferCopy& copy : copies) {
            src.MarkUsage(copy.src_offset, copy.size);
            EncodeCopy(src.Handle(), copy.src_offset, dst, copy.dst_offset, copy.size);
        }
        return;
    }
    if (wait_tick != 0) {
        // Waits only for the writes this download depends on. The current command buffer is
        // only submitted when they were recorded in it.
        scheduler.Wait(wait_tick, Scheduler::WaitReason::BufferDownload);
    }
    // The shared allocation holds the latest data, the download needs no GPU copy. The buffer
    // cache writes the bytes to guest memory right after, which then holds them too.
    u8* const dst_pointer = static_cast<u8*>(dst.contents);
    for (const VideoCommon::BufferCopy& copy : copies) {
        std::memcpy(dst_pointer + copy.dst_offset, src.Mapped() + copy.src_offset, copy.size);
        src.ClearProducedBits(copy.src_offset, copy.size);
    }
}

void BufferCacheRuntime::ClearBuffer(Buffer& dst, u32 offset, size_t size, u32 value) {
    if (!dst.Handle() || size == 0) {
        return;
    }
    ASSERT(offset + size <= dst.SizeBytes());
    const u8 first_byte = static_cast<u8>(value);
    const bool uniform_bytes = value == first_byte * 0x01010101U;
    const auto fill_pattern = [value](u8* pointer, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            pointer[i] = static_cast<u8>(value >> ((i % 4) * 8));
        }
    };
    dst.ApplyOverlaysOverlapping(offset, size);
    if (!dst.IsRegionUsed(offset, size)) {
        if (uniform_bytes) {
            std::memset(dst.Mapped() + offset, first_byte, size);
        } else {
            fill_pattern(dst.Mapped() + offset, size);
        }
        return;
    }
    dst.MarkGpuWrite(offset, size, Buffer::GpuWriteKind::Produced);
    scheduler.CountBufferFill();
    if (uniform_bytes && IsBlitAligned(offset, 0, size)) {
        [scheduler.BlitEncoder() fillBuffer:dst.Handle()
                                      range:NSMakeRange(offset, size)
                                      value:first_byte];
        return;
    }
    const StagingBufferRef staging = staging_pool.Request(size, MemoryUsage::Upload);
    fill_pattern(staging.mapped_span.data(), size);
    EncodeCopy(staging.buffer, staging.offset, dst.Handle(), offset, size);
}

const u8* BufferCacheRuntime::ReadForCpu(Buffer& buffer, u64 offset, u64 size) {
    if (!buffer.Handle() || offset + size > buffer.SizeBytes()) {
        return nullptr;
    }
    u64 wait_tick = 0;
    const u8* const bytes = buffer.ReadableBytes(offset, size, wait_tick);
    if (wait_tick != 0) {
        // TODO: Convert on the GPU with a compute pass to avoid this stall.
        scheduler.Wait(wait_tick, Scheduler::WaitReason::CpuRead);
    }
    return bytes;
}

void BufferCacheRuntime::BindConvertedIndices(std::span<const u32> indices) {
    if (indices.empty()) {
        index_binding = IndexBufferBinding{};
        return;
    }
    const StagingBufferRef ref = staging_pool.RequestStream(indices.size_bytes(), 4);
    std::memcpy(ref.mapped_span.data(), indices.data(), indices.size_bytes());
    index_binding = IndexBufferBinding{
        .buffer = ref.buffer,
        .offset = ref.offset,
        .count = static_cast<u32>(indices.size()),
        .type = MTLIndexTypeUInt32,
        .converted_quads = true,
    };
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 first, u32 count, Buffer& buffer, u32 offset,
                                         u32 size) {
    const u32 index_size = IndexSize(index_format);
    const u64 start = static_cast<u64>(offset) + static_cast<u64>(first) * index_size;
    const bool is_quads =
        topology == PrimitiveTopology::Quads || topology == PrimitiveTopology::QuadStrip;
    const bool needs_cpu_path = is_quads || index_size == 1 || start % BLIT_ALIGNMENT != 0;
    if (!needs_cpu_path) {
        const BufferBinding binding = buffer.Resolve(start, static_cast<u64>(count) * index_size);
        index_binding = IndexBufferBinding{
            .buffer = binding.buffer,
            .offset = binding.offset,
            .count = count,
            .type = index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32,
            .converted_quads = false,
        };
        return;
    }
    // Metal has no 8-bit indices nor quads, and needs aligned index offsets: widen on the CPU.
    const u64 bytes = static_cast<u64>(count) * index_size;
    if (start + bytes > static_cast<u64>(offset) + size) {
        LOG_WARNING(Render_Metal, "Index range exceeds the bound index buffer");
    }
    const u8* const indices = ReadForCpu(buffer, start, bytes);
    if (!indices) {
        index_binding = IndexBufferBinding{};
        return;
    }
    index_scratch.clear();
    if (is_quads) {
        ForEachQuadTriangleVertex(topology == PrimitiveTopology::QuadStrip, count, [&](u32 index) {
            index_scratch.push_back(ReadIndex(indices, index_size, index));
        });
    } else {
        index_scratch.resize(count);
        for (u32 i = 0; i < count; ++i) {
            index_scratch[i] = ReadIndex(indices, index_size, i);
        }
    }
    BindConvertedIndices(index_scratch);
    index_binding.converted_quads = is_quads;
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    index_scratch.clear();
    ForEachQuadTriangleVertex(topology == PrimitiveTopology::QuadStrip, count,
                              [&](u32 index) { index_scratch.push_back(first + index); });
    BindConvertedIndices(index_scratch);
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, id<MTLBuffer> buffer, u32 offset, u32 size,
                                          u32 stride) {
    if (index >= vertex_buffers.size()) {
        return;
    }
    vertex_buffers[index] = BufferBinding{.buffer = buffer, .offset = offset, .size = size};
    vertex_strides[index] = stride;
    vertex_buffer_count = (std::max)(vertex_buffer_count, index + 1);
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    for (size_t i = 0; i < bindings.buffers.size(); ++i) {
        const size_t index = bindings.min_index + i;
        if (index >= vertex_buffers.size()) {
            break;
        }
        Buffer* const buffer = bindings.buffers[i];
        if (buffer && buffer->Handle()) {
            vertex_buffers[index] = buffer->Resolve(bindings.offsets[i], bindings.sizes[i]);
        } else {
            vertex_buffers[index] = BufferBinding{};
        }
        vertex_strides[index] = bindings.strides[i];
    }
    vertex_buffer_count = (std::max)(vertex_buffer_count, bindings.max_index);
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    static bool logged = false;
    if (!logged && !bindings.buffers.empty()) {
        LOG_WARNING(Render_Metal, "Transform feedback is not natively supported by Metal");
        logged = true;
    }
    for (size_t i = 0; i < bindings.buffers.size(); ++i) {
        const size_t index = bindings.min_index + i;
        if (index >= transform_feedback.size()) {
            break;
        }
        Buffer* const buffer = bindings.buffers[i];
        if (buffer) {
            buffer->ApplyOverlaysOverlapping(bindings.offsets[i], bindings.sizes[i]);
            buffer->MarkGpuWrite(bindings.offsets[i], bindings.sizes[i],
                                 Buffer::GpuWriteKind::Produced);
        }
        transform_feedback[index] = BufferBinding{
            .buffer = buffer ? buffer->Handle() : nil,
            .offset = bindings.offsets[i],
            .size = bindings.sizes[i],
        };
    }
}

std::span<u8> BufferCacheRuntime::BindMappedUniformBuffer(size_t stage, u32 binding_index,
                                                          u32 size) {
    const StagingBufferRef ref = staging_pool.RequestStream(size, STREAM_UNIFORM_ALIGNMENT);
    if (stage < stages.size() && binding_index < stages[stage].uniforms.size()) {
        stages[stage].uniforms[binding_index] = BufferBinding{
            .buffer = ref.buffer,
            .offset = ref.offset,
            .size = size,
        };
    }
    return ref.mapped_span;
}

void BufferCacheRuntime::BindUniformBuffer(size_t stage, u32 binding_index, Buffer& buffer,
                                           u32 offset, u32 size) {
    if (stage >= stages.size() || binding_index >= stages[stage].uniforms.size()) {
        return;
    }
    stages[stage].uniforms[binding_index] = buffer.Resolve(offset, size);
}

void BufferCacheRuntime::BindComputeUniformBuffer(u32 binding_index, Buffer& buffer, u32 offset,
                                                  u32 size) {
    if (binding_index >= compute_uniforms.size()) {
        return;
    }
    compute_uniforms[binding_index] = buffer.Resolve(offset, size);
}

void BufferCacheRuntime::BindStorageBuffer(size_t stage, u32 binding_index, Buffer& buffer,
                                           u32 offset, u32 size, bool is_written) {
    if (stage >= stages.size() || binding_index >= stages[stage].storages.size()) {
        return;
    }
    // The buffer cache marked the range right before this call.
    buffer.SoftenLastUsage();
    if (is_written) {
        buffer.MarkGpuWrite(offset, size, Buffer::GpuWriteKind::Produced);
        buffer.SoftenLastUsage();
    }
    stages[stage].storages[binding_index] = buffer.Resolve(offset, size, is_written);
}

void BufferCacheRuntime::BindComputeStorageBuffer(u32 binding_index, Buffer& buffer, u32 offset,
                                                  u32 size, bool is_written) {
    if (binding_index >= compute_storages.size()) {
        return;
    }
    buffer.SoftenLastUsage();
    if (is_written) {
        buffer.MarkGpuWrite(offset, size, Buffer::GpuWriteKind::Produced);
        buffer.SoftenLastUsage();
    }
    compute_storages[binding_index] = buffer.Resolve(offset, size, is_written);
}

void BufferCacheRuntime::BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                                           PixelFormat format) {
    texture_buffers.push_back(buffer.ResolveView(offset, size, format, false));
}

void BufferCacheRuntime::BindImageBuffer(Buffer& buffer, u32 offset, u32 size,
                                         PixelFormat format) {
    // Bound in the same list as texture buffers, the buffer cache calls both in binding order.
    buffer.MarkGpuWrite(offset, size, Buffer::GpuWriteKind::Produced);
    texture_buffers.push_back(buffer.ResolveView(offset, size, format, true));
}

void BufferCacheRuntime::ApplyVertexBuffers(id<MTLRenderCommandEncoder> encoder, u32 first_index,
                                            std::span<AppliedVertexBuffer> applied) const {
    for (u32 i = 0; i < vertex_buffer_count; ++i) {
        const u32 index = first_index + i;
        if (index >= MAX_METAL_BUFFER_BINDINGS) {
            LOG_WARNING(Render_Metal, "Vertex buffer {} exceeds the Metal binding table", i);
            break;
        }
        const BufferBinding& binding = vertex_buffers[i];
        if (i < applied.size()) {
            AppliedVertexBuffer& slot = applied[i];
            if (slot.buffer == binding.buffer && slot.offset == binding.offset) {
                continue;
            }
            slot.buffer = binding.buffer;
            slot.offset = binding.offset;
        }
        [encoder setVertexBuffer:binding.buffer offset:binding.offset atIndex:index];
    }
}

void BufferCacheRuntime::ApplyStageBuffers(id<MTLRenderCommandEncoder> encoder, size_t stage,
                                           u32 uniform_base, u32 storage_base) const {
    const bool is_vertex = stage == 0;
    const bool is_fragment = stage == 4;
    if (!is_vertex && !is_fragment) {
        return;
    }
    const auto bind = [&](const BufferBinding& binding, u32 index) {
        if (!binding.buffer || index >= MAX_METAL_BUFFER_BINDINGS) {
            return;
        }
        if (is_vertex) {
            [encoder setVertexBuffer:binding.buffer offset:binding.offset atIndex:index];
        } else {
            [encoder setFragmentBuffer:binding.buffer offset:binding.offset atIndex:index];
        }
    };
    const StageBindings& bindings = stages[stage];
    for (u32 i = 0; i < bindings.uniforms.size(); ++i) {
        bind(bindings.uniforms[i], uniform_base + i);
    }
    for (u32 i = 0; i < bindings.storages.size(); ++i) {
        bind(bindings.storages[i], storage_base + i);
    }
}

void BufferCacheRuntime::ApplyComputeBuffers(id<MTLComputeCommandEncoder> encoder,
                                             u32 uniform_base, u32 storage_base) const {
    const auto bind = [&](const BufferBinding& binding, u32 index) {
        if (binding.buffer && index < MAX_METAL_BUFFER_BINDINGS) {
            [encoder setBuffer:binding.buffer offset:binding.offset atIndex:index];
        }
    };
    for (u32 i = 0; i < compute_uniforms.size(); ++i) {
        bind(compute_uniforms[i], uniform_base + i);
    }
    for (u32 i = 0; i < compute_storages.size(); ++i) {
        bind(compute_storages[i], storage_base + i);
    }
}

} // namespace Metal

namespace VideoCommon {
template class VideoCommon::BufferCache<Metal::BufferCacheParams>;
}
