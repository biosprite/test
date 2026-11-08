// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "mtl_buffer_cache.h must be included from Objective-C++ compiled with -fobjc-arc"
#endif

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#import <Metal/Metal.h>

#include "common/common_types.h"
#include "common/slot_vector.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/surface.h"

namespace Metal {

class Device;

/// Records the cache command buffers and tracks their completion with monotonic ticks.
/// Every resource used by recorded commands is tagged with CurrentTick(); it is free again once
/// IsFree() returns true for that tick. Not thread safe: like the Vulkan scheduler it must only be
/// driven from the GPU thread. Completion handlers only touch the atomic tick.
class Scheduler {
public:
    explicit Scheduler(const Device& device);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Returns the command buffer being recorded, creating it on demand.
    [[nodiscard]] id<MTLCommandBuffer> CommandBuffer();

    /// What ended a render pass. Kept with the key of the pass, so that a pass restarted on
    /// the same framebuffer right after is attributed to the break in between.
    enum class PassEnd : u8 {
        /// A blit encoder opened for buffer copies and fills.
        BufferBlit,
        /// A blit encoder opened for image uploads, downloads and copies.
        ImageBlit,
        Compute,
        /// A deferred clear ran as a pass of its own.
        Clear,
        /// A blit drawn as a pass of its own (scaled blits, format conversions).
        BlitPass,
        /// Shaders of the pass wrote memory that a barrier orders.
        Barrier,
        /// A draw samples the depth buffer the pass wrote.
        FeedbackLoop,
        /// Command buffer submission.
        Flush,
        /// Pass switches and explicit ends by copies.
        Other,
    };

    /// Returns a blit encoder for the current command buffer, ending any other open encoder.
    /// reason tells what the blit is for when it ends a render pass.
    [[nodiscard]] id<MTLBlitCommandEncoder> BlitEncoder(PassEnd reason = PassEnd::BufferBlit);

    /// Incremented every time a blit encoder is created; resources tag themselves with it to
    /// tell whether the open blit encoder already touched them.
    [[nodiscard]] u64 BlitGeneration() const noexcept {
        return blit_generation;
    }

    /// Blit encoder for commands touching the resources tagged with the generations. When the
    /// open blit encoder already touched one of them a new encoder keeps the commands ordered;
    /// otherwise consecutive blits on different resources share one encoder. The tags are
    /// updated to the encoder returned.
    [[nodiscard]] id<MTLBlitCommandEncoder> BlitEncoderFor(u64& first_tag, u64& second_tag);
    [[nodiscard]] id<MTLBlitCommandEncoder> BlitEncoderFor(u64& tag) {
        return BlitEncoderFor(tag, tag);
    }

    /// Records a buffer copy for the side command buffer of the current tick. The side buffer
    /// is committed right before the current command buffer, so its copies run after every
    /// earlier submission and before anything recorded in the current one, without touching
    /// the open encoder. Only for data no command of the current tick has used yet. The copies
    /// are encoded in order at submission; a copy continuing the previous one (both ranges
    /// contiguous, same buffers) is merged into it, so uploads of consecutive ranges cost one
    /// blit.
    void SideCopy(id<MTLBuffer> src, u64 src_offset, id<MTLBuffer> dst, u64 dst_offset, u64 size);

    /// Returns the blit encoder of the side command buffer for texture commands, see SideCopy
    /// for what the side buffer is. Only for textures no command of the current tick has used
    /// yet: those the caches bound are tracked by the images, those touched without an image
    /// (deferred clears, view blits) by MarkTextureUsed.
    [[nodiscard]] id<MTLBlitCommandEncoder> SideBlitEncoder();

    /// Records that a command of the current command buffer touched the texture (any view of
    /// it), so that no side command targets it until the next submission.
    void MarkTextureUsed(id<MTLTexture> texture);

    [[nodiscard]] bool IsTextureUsedThisTick(id<MTLTexture> texture) const;

    /// Returns the render encoder of the pass identified by key, starting a new pass when a
    /// different one (or no pass) is open. A non-nil descriptor always starts a new pass with
    /// it (e.g. to clear attachments through load actions), later requests with the same key
    /// keep using that pass.
    [[nodiscard]] id<MTLRenderCommandEncoder> RenderEncoder(MTLRenderPassDescriptor* key,
                                                            MTLRenderPassDescriptor* descriptor = nil);

    /// Returns a compute encoder for the current command buffer, ending any other open encoder.
    [[nodiscard]] id<MTLComputeCommandEncoder> ComputeEncoder();

    /// True when a render pass for key is open.
    [[nodiscard]] bool IsRenderPassActive(MTLRenderPassDescriptor* key) const noexcept {
        return render_encoder != nil && render_pass_key == key;
    }

    /// Incremented every time a render or compute encoder is created, encoder state set by
    /// callers must be reapplied when it changes.
    [[nodiscard]] u64 EncoderGeneration() const noexcept {
        return encoder_generation;
    }

    /// Ends the open encoder, the next command starts a new one. Metal tracks hazards between
    /// encoders, so this works as a full barrier for tracked resources. reason attributes the
    /// restart if the next pass starts on the same framebuffer.
    void EndEncoding(PassEnd reason = PassEnd::Other);

    /// Ends the open blit encoder, if any, leaving render and compute encoders open.
    void EndBlitEncoding();

    /// True when a render pass is open on the current command buffer.
    [[nodiscard]] bool HasRenderPass() const noexcept {
        return render_encoder != nil;
    }

    /// Commits the current command buffer, even when empty, so that CurrentTick() gets signaled.
    void Flush();

    /// Flushes and waits for every submitted command buffer.
    void Finish();

    /// Why the GPU thread blocks on the GPU, for the statistics.
    enum class WaitReason {
        /// A synchronous buffer download of a region the GPU wrote.
        BufferDownload,
        /// Guest indices converted on the CPU from a region the GPU wrote.
        CpuRead,
        ImageDownload,
        UnalignedCopy,
        /// Explicit synchronization requested by the common caches or at shutdown.
        Sync,
    };

    /// Waits until the given tick has been completed by the GPU, flushing if needed.
    void Wait(u64 tick, WaitReason reason = WaitReason::Sync);

    /// Waits for an already submitted tick without flushing. Safe to call from any thread.
    void WaitSubmitted(u64 tick);

    /// Runs func on a Metal thread once the current command buffer completes, before its tick
    /// is published as completed.
    void OnCompletion(std::function<void()>&& func);

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick;
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return gpu_tick.load(std::memory_order_acquire) >= tick;
    }

    /// True when nothing has been recorded in the current command buffer.
    [[nodiscard]] bool IsEmpty() const noexcept {
        return command_buffer == nil && side_command_buffer == nil;
    }

    /// GPU execution time of the command buffers completed since the last call, and how long
    /// after their commit they started: the lag of the GPU behind the GPU thread.
    struct GpuTime {
        u64 busy_ns = 0;
        u64 lag_ns = 0;
        u64 command_buffers = 0;
    };
    [[nodiscard]] GpuTime TakeGpuTime() noexcept {
        return GpuTime{
            .busy_ns = gpu_busy_ns.exchange(0, std::memory_order_relaxed),
            .lag_ns = gpu_lag_ns.exchange(0, std::memory_order_relaxed),
            .command_buffers = gpu_timed_buffers.exchange(0, std::memory_order_relaxed),
        };
    }

    /// Running totals of the work recorded, for the periodic statistics log.
    struct Stats {
        u64 command_buffers = 0;
        u64 render_passes = 0;
        /// Render passes started on the framebuffer of the pass ended right before them: the
        /// break in between (a copy, a barrier, a feedback loop) split one pass in two. Split
        /// by what ended the earlier pass, see PassEnd.
        u64 pass_restarts = 0;
        u64 restarts_buffer_blit = 0;
        u64 restarts_image_blit = 0;
        u64 restarts_compute = 0;
        u64 restarts_clear = 0;
        u64 restarts_blit_pass = 0;
        u64 restarts_barrier = 0;
        u64 restarts_feedback_loop = 0;
        u64 restarts_flush = 0;
        u64 restarts_other = 0;
        u64 blit_encoders = 0;
        u64 compute_encoders = 0;
        /// Render passes ended because a draw samples the depth buffer written in them.
        u64 feedback_loop_breaks = 0;
        /// Buffer uploads copied on the GPU right away, split by why they could not wait for
        /// the end of the encoder: the copy was not 4 byte aligned, or the buffer already
        /// held the maximum number of deferred uploads.
        u64 buffer_upload_blits = 0;
        u64 upload_blits_unaligned = 0;
        u64 upload_blits_capped = 0;
        /// Buffer uploads deferred to the end of the open render pass.
        u64 deferred_uploads = 0;
        /// Deferred uploads merged with an earlier one of the same pass (same range, a sub
        /// range or a range covering it) without touching the pass.
        u64 overlay_merges = 0;
        /// Deferred uploads grown to cover a binding or an upload straddling them, with the
        /// buffer bytes in between, instead of being applied early.
        u64 overlay_extensions = 0;
        /// Uploads to regions only earlier submissions use, copied by the side command buffer
        /// (no staging binding, no pass break), and rewrites patched into their staging. The
        /// blits are the copies actually encoded after merging contiguous ones; gap bytes are
        /// the buffer bytes copied along to bridge two uploads into one blit.
        u64 side_copies = 0;
        u64 side_patches = 0;
        u64 side_blits = 0;
        u64 side_gap_bytes = 0;
        u64 side_command_buffers = 0;
        /// Image uploads and copies recorded in the side command buffer because no command of
        /// the tick had touched the images, so they split no render pass.
        u64 side_image_uploads = 0;
        u64 side_image_copies = 0;
        /// Deferred uploads applied early because a GPU write, a copy or a binding straddling
        /// them needed the buffer itself; each ends the open render pass.
        u64 overlay_applies = 0;
        /// GPU copies feeding synchronous buffer downloads and buffer fills.
        u64 buffer_download_copies = 0;
        u64 buffer_fills = 0;
        u64 image_uploads = 0;
        u64 image_downloads = 0;
        /// Image to image copies of the texture cache (mostly alias synchronization), and how
        /// many of them were between images that could share one allocation.
        u64 image_copies = 0;
        u64 shareable_image_copies = 0;
        /// Fermi2D copies done with blit encoders.
        u64 surface_copies = 0;
        /// Full framebuffer clears recorded as load actions of a later pass: deferred when
        /// requested, applied when a pass on that framebuffer started, flushed when something
        /// else needed them first and they ran as a pass of their own.
        u64 deferred_clears = 0;
        u64 applied_clears = 0;
        u64 flushed_clears = 0;
        /// Times the GPU thread blocked on the GPU and the total time spent blocked, with the
        /// count split by WaitReason.
        u64 gpu_waits = 0;
        u64 gpu_wait_ns = 0;
        u64 waits_buffer_download = 0;
        u64 waits_cpu_read = 0;
        u64 waits_image_download = 0;
        u64 waits_unaligned_copy = 0;
        u64 waits_sync = 0;
        /// Why deferred clears ran as passes of their own: evicted from the pending list, a
        /// texture cache operation or a bound view needed the texture, a pass on another
        /// framebuffer sharing a texture started, or the work was flushed for presentation.
        u64 clears_run_evicted = 0;
        u64 clears_run_image_op = 0;
        u64 clears_run_bound_view = 0;
        u64 clears_run_overlap = 0;
        u64 clears_run_flush = 0;
        /// Render passes ended without storing their depth, or stencil, attachment back to
        /// memory because nothing in the pass wrote it.
        u64 depth_dontcare_stores = 0;
        u64 stencil_dontcare_stores = 0;
    };

    /// Why a deferred clear has to run before the pass that could have folded it.
    enum class ClearFlushReason {
        Evicted,
        ImageOperation,
        BoundView,
        PassOverlap,
        Flush,
    };

    [[nodiscard]] const Stats& GetStats() const noexcept {
        return stats;
    }

    enum class UploadBlitReason {
        Unaligned,
        TooManyOverlays,
    };

    void CountFeedbackLoopBreak() noexcept {
        ++stats.feedback_loop_breaks;
    }
    void CountBufferUploadBlit(UploadBlitReason reason) noexcept {
        ++stats.buffer_upload_blits;
        switch (reason) {
        case UploadBlitReason::Unaligned:
            ++stats.upload_blits_unaligned;
            break;
        case UploadBlitReason::TooManyOverlays:
            ++stats.upload_blits_capped;
            break;
        }
    }
    void CountDeferredUpload() noexcept {
        ++stats.deferred_uploads;
    }
    void CountOverlayMerge() noexcept {
        ++stats.overlay_merges;
    }
    void CountOverlayExtension() noexcept {
        ++stats.overlay_extensions;
    }
    void CountSideCopy() noexcept {
        ++stats.side_copies;
    }
    void CountSidePatch() noexcept {
        ++stats.side_patches;
    }
    void CountSideGapBytes(u64 bytes) noexcept {
        stats.side_gap_bytes += bytes;
    }
    void CountSideImageUpload() noexcept {
        ++stats.side_image_uploads;
    }
    void CountSideImageCopy() noexcept {
        ++stats.side_image_copies;
    }
    void CountOverlayApply() noexcept {
        ++stats.overlay_applies;
    }
    void CountImageUpload() noexcept {
        ++stats.image_uploads;
    }
    void CountImageDownload() noexcept {
        ++stats.image_downloads;
    }
    void CountImageCopy(bool shareable) noexcept {
        ++stats.image_copies;
        if (shareable) {
            ++stats.shareable_image_copies;
        }
    }
    void CountSurfaceCopy() noexcept {
        ++stats.surface_copies;
    }
    void CountBufferDownloadCopy() noexcept {
        ++stats.buffer_download_copies;
    }
    void CountBufferFill() noexcept {
        ++stats.buffer_fills;
    }

    /// Records that a draw of the open render pass writes depth, or stencil.
    void MarkPassDepthWrite() noexcept {
        depth_write_generation = encoder_generation;
    }
    void MarkPassStencilWrite() noexcept {
        stencil_write_generation = encoder_generation;
    }

    /// True when a draw of the open render pass wrote depth, or stencil.
    [[nodiscard]] bool PassWroteDepth() const noexcept {
        return render_encoder != nil && depth_write_generation == encoder_generation;
    }
    [[nodiscard]] bool PassWroteStencil() const noexcept {
        return render_encoder != nil && stencil_write_generation == encoder_generation;
    }

    /// The texture cache reports draws sampling the depth buffer they render with; the
    /// rasterizer resolves the request with the aspects the draw samples.
    void RequestFeedbackLoopCheck() noexcept {
        feedback_loop_requested = true;
    }
    [[nodiscard]] bool TakeFeedbackLoopRequest() noexcept {
        return std::exchange(feedback_loop_requested, false);
    }

    /// Records a clear of every attachment selected in descriptor (a copy of key with clear
    /// load actions) without starting a pass. The clear becomes the load action of the next
    /// pass started on key, which costs nothing on tile based GPUs; it runs as a pass of its
    /// own before any command that could read or write the cleared textures.
    void DeferClear(MTLRenderPassDescriptor* key, MTLRenderPassDescriptor* descriptor,
                    bool clears_depth, bool clears_stencil);

    [[nodiscard]] bool HasPendingClears() const noexcept {
        return !pending_clears.empty();
    }

    /// True when a deferred clear targets the texture (any view of it).
    [[nodiscard]] bool HasPendingClearFor(id<MTLTexture> texture) const;

    /// Runs every deferred clear now, ending the open encoder.
    void FlushPendingClears(ClearFlushReason reason);

    /// Runs the deferred clears targeting any of the textures, ending the open encoder if
    /// there are any.
    void FlushPendingClearsFor(std::initializer_list<id<MTLTexture>> textures,
                               ClearFlushReason reason);

    /// Called right after a render or compute encoder is ended and before the next encoder
    /// starts, and when the command buffer is submitted, so that work waiting for an encoder
    /// to end (deferred buffer uploads) is encoded in between.
    void SetEncoderEndCallback(std::function<void()> callback) {
        on_encoder_end = std::move(callback);
    }

private:
    /// Games clear dozens of framebuffers ahead of drawing to them (reflection probes clear
    /// every face and mip level); evicting one early costs an extra pass, so the list is
    /// scanned linearly and kept generous.
    static constexpr size_t MAX_PENDING_CLEARS = 256;
    /// Color attachments, depth and stencil.
    static constexpr size_t MAX_PASS_TEXTURES = 10;

    struct PendingClear {
        MTLRenderPassDescriptor* key;
        MTLRenderPassDescriptor* descriptor;
        /// Textures the attachments are views of.
        std::array<id<MTLTexture>, MAX_PASS_TEXTURES> textures;
        size_t num_textures = 0;
        bool clears_depth = false;
        bool clears_stencil = false;
    };

    /// A buffer copy waiting for the side command buffer, see SideCopy.
    struct SideCopyCommand {
        id<MTLBuffer> src;
        id<MTLBuffer> dst;
        u64 src_offset;
        u64 dst_offset;
        u64 size;
    };

    void SignalTick(u64 tick);

    /// Returns the side command buffer of the current tick, creating it on demand.
    [[nodiscard]] id<MTLCommandBuffer> SideCommandBuffer();

    /// Encodes the recorded side copies, before the side command buffer is committed.
    void EncodeSideCopies();

    void CountRestart(PassEnd reason) noexcept;

    /// Textures behind the attachments of a render pass descriptor.
    static size_t PassTextures(MTLRenderPassDescriptor* descriptor,
                               std::array<id<MTLTexture>, MAX_PASS_TEXTURES>& textures);

    /// Encodes the clear as a pass without draws. The caller ended the open encoder.
    void ExecutePendingClear(const PendingClear& clear, ClearFlushReason reason);

    /// Runs the deferred clears sharing a texture with the pass about to start on descriptor.
    void FlushPendingClearsOverlapping(MTLRenderPassDescriptor* descriptor);

    void FlushPendingClearsMatching(std::span<const id<MTLTexture>> roots,
                                    ClearFlushReason reason);

    /// Runs the encoder end callback once, closing the blit encoder it may have opened.
    void RunEncoderEndCallback();

    /// Sets the store actions the descriptor of the open pass left unknown: an aspect no draw
    /// of the pass wrote is not stored back, its memory already holds what was loaded.
    void FinalizeStoreActions();

    Stats stats;
    u64 depth_write_generation = ~u64{0};
    u64 stencil_write_generation = ~u64{0};
    /// The open pass started with an unknown depth, or stencil, store action.
    bool depth_store_pending = false;
    bool stencil_store_pending = false;
    bool feedback_loop_requested = false;
    std::function<void()> on_encoder_end;
    bool in_encoder_end_callback = false;
    std::vector<PendingClear> pending_clears;

    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command_buffer;
    id<MTLCommandBuffer> side_command_buffer;
    id<MTLBlitCommandEncoder> side_blit_encoder;
    std::vector<SideCopyCommand> side_copies;
    /// Root textures that commands of the current command buffer touched without an image
    /// to record it on, cleared at submission.
    std::vector<id<MTLTexture>> tick_textures;
    std::atomic<u64> gpu_busy_ns{};
    std::atomic<u64> gpu_lag_ns{};
    std::atomic<u64> gpu_timed_buffers{};
    /// End of the GPU time already counted as busy, in the host time base.
    std::mutex gpu_time_mutex;
    double gpu_timeline_end = 0.0;
    id<MTLBlitCommandEncoder> blit_encoder;
    id<MTLRenderCommandEncoder> render_encoder;
    id<MTLComputeCommandEncoder> compute_encoder;
    /// Strong reference, so the identity of the key can't be reused while the pass is open.
    MTLRenderPassDescriptor* render_pass_key;
    /// Key of the last render pass ended, kept alive so a new pass can be compared with it,
    /// and what ended it.
    MTLRenderPassDescriptor* last_ended_pass_key;
    PassEnd last_pass_end = PassEnd::Other;
    u64 encoder_generation = 0;
    u64 blit_generation = 0;
    std::shared_ptr<std::vector<std::function<void()>>> pending_callbacks;

    u64 current_tick = 1;
    std::atomic<u64> gpu_tick{0};
    std::mutex tick_mutex;
    std::condition_variable tick_cv;
};

enum class MemoryUsage {
    Upload,
    Download,
};

/// CPU visible memory handed out by StagingBufferPool. With unified memory every MTLBuffer used
/// by the caches is StorageModeShared, so mapped_span aliases buffer.contents directly.
struct StagingBufferRef {
    id<MTLBuffer> buffer;
    size_t offset = 0;
    std::span<u8> mapped_span;
    MemoryUsage usage = MemoryUsage::Upload;
    u32 log2_level = 0;
    size_t index = 0;
    bool deferred = false;
};

class StagingBufferPool {
public:
    explicit StagingBufferPool(const Device& device, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// Returns a buffer of at least size bytes, used by commands of the current tick. Deferred
    /// buffers are not recycled until FreeDeferred is called on them.
    [[nodiscard]] StagingBufferRef Request(size_t size, MemoryUsage usage, bool deferred = false);

    void FreeDeferred(StagingBufferRef& ref);

    /// Sub-allocates short lived upload memory (uniform data, generated indices) from a ring of
    /// shared buffers.
    [[nodiscard]] StagingBufferRef RequestStream(size_t size, size_t alignment);

    /// Sub-allocates staging memory for the side copies of the current tick from a ring of
    /// its own, so that consecutive side uploads are contiguous in staging memory and their
    /// copies can be merged. Blit aligned.
    [[nodiscard]] StagingBufferRef RequestSideStream(size_t size);

    void TickFrame();

    /// Bytes of staging memory currently allocated, in use or cached for reuse.
    [[nodiscard]] u64 AllocatedBytes() const noexcept {
        return allocated_bytes;
    }

private:
    static constexpr u32 NUM_LEVELS = 48;
    static constexpr u32 MIN_LOG2 = 12;
    static constexpr size_t STREAM_CHUNK_SIZE = 4 * 1024 * 1024;

    struct Entry {
        id<MTLBuffer> buffer;
        std::span<u8> mapped_span;
        u64 tick = 0;
        u64 frame = 0;
        bool deferred = false;
    };

    struct Level {
        std::vector<Entry> entries;
        size_t iterate_index = 0;
    };

    using Cache = std::array<Level, NUM_LEVELS>;

    /// A ring of upload chunks sub-allocated linearly.
    struct Stream {
        std::optional<StagingBufferRef> chunk;
        size_t offset = 0;
    };

    Cache& CacheFor(MemoryUsage usage) noexcept {
        return usage == MemoryUsage::Upload ? upload_cache : download_cache;
    }

    std::optional<StagingBufferRef> TryReuse(MemoryUsage usage, u32 log2, bool deferred);
    StagingBufferRef Create(MemoryUsage usage, u32 log2, bool deferred);
    void ReleaseUnused(Cache& cache);
    StagingBufferRef RequestFromStream(Stream& stream, size_t size, size_t alignment);

    id<MTLDevice> device;
    Scheduler& scheduler;

    Cache upload_cache;
    Cache download_cache;
    u64 allocated_bytes = 0;
    u64 frame = 0;

    Stream stream;
    Stream side_stream;
};

class BufferCacheRuntime;

/// Vertex buffer last bound to a slot of a render command encoder.
struct AppliedVertexBuffer {
    __unsafe_unretained id<MTLBuffer> buffer = nil;
    u64 offset = 0;
};

/// Buffer bindings recorded by the buffer cache, applied to encoders by the pipelines.
struct BufferBinding {
    id<MTLBuffer> buffer;
    u64 offset = 0;
    u64 size = 0;
};

class Buffer : public VideoCommon::BufferBase {
public:
    explicit Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params);
    explicit Buffer(BufferCacheRuntime& runtime, DAddr cpu_addr, u64 size_bytes,
                    bool sparse_compatible);

    /// Returns a texture buffer view of the given range, nil when Metal cannot alias it.
    [[nodiscard]] id<MTLTexture> View(u32 offset, u32 size, VideoCore::Surface::PixelFormat format);

    /// Texture buffer view for a binding: a view of the staging copy when a deferred upload
    /// covers the range (shader writes are recorded on it like Resolve does), otherwise a view
    /// of the buffer after applying the overlapping uploads.
    [[nodiscard]] id<MTLTexture> ResolveView(u32 offset, u32 size,
                                             VideoCore::Surface::PixelFormat format,
                                             bool is_written);

    [[nodiscard]] id<MTLBuffer> Handle() const noexcept {
        return buffer;
    }

    [[nodiscard]] u8* Mapped() const noexcept {
        return mapped;
    }

    /// Writes guest data into the buffer. The bytes are copied straight into the shared
    /// allocation when no pending GPU work touches the region. Otherwise they are staged: the
    /// copy into the buffer is deferred to the end of the open encoder (or the submission) and
    /// the draws and dispatches until then read the staging copy through Resolve(), which keeps
    /// render passes and compute encoders whole.
    void ImmediateUpload(size_t offset, std::span<const u8> data);

    /// Binding of [offset, size) for GPU reads: the staging copy of a deferred upload covering
    /// the range, otherwise the buffer. A partially covered range applies the deferred uploads
    /// first. With is_written the shaders write the binding: a staging copy records the write,
    /// the copy at the end of the encoder then carries the shader output into the buffer.
    [[nodiscard]] BufferBinding Resolve(u64 offset, u64 size, bool is_written = false);

    /// Applies the deferred uploads now when any overlaps [offset, size): copies and texture
    /// views must see the buffer itself.
    void ApplyOverlaysOverlapping(u64 offset, u64 size);

    /// CPU readable bytes of [offset, size), from a deferred upload when one holds them.
    /// wait_tick is the tick of GPU writes still pending on those bytes, zero when none.
    [[nodiscard]] const u8* ReadableBytes(u64 offset, u64 size, u64& wait_tick);

    /// Records that GPU commands of the current tick access [offset, offset + size).
    void MarkUsage(u64 offset, u64 size) noexcept;

    /// What a pending GPU write puts in the buffer: data the CPU wrote (guest memory holds it
    /// too) or data produced on the GPU (shaders, copies, fills), which guest memory lacks
    /// until a download.
    enum class GpuWriteKind {
        Upload,
        Produced,
    };

    /// Records that GPU commands of the current tick may write [offset, offset + size).
    void MarkGpuWrite(u64 offset, u64 size, GpuWriteKind kind) noexcept;

    /// True when GPU work that has not completed yet may access the region.
    [[nodiscard]] bool IsRegionUsed(u64 offset, u64 size) const noexcept;

    /// True when commands recorded in the current, unsubmitted tick access the region through
    /// a binding that reads exactly what it covers (vertex, index, uniform, texel buffers).
    [[nodiscard]] bool IsRegionUsedByCurrentTick(u64 offset, u64 size) const noexcept;

    /// Downgrades the granules the last MarkUsage call newly marked in this tick to soft
    /// uses: they still keep CPU writes off the shared allocation until the GPU is done, but
    /// a CPU write no longer has to be ordered inside the tick. Storage buffer bindings cover
    /// whole pools while the shaders touch one slice, and a game only rewrites a slice its own
    /// fences proved the GPU finished with; treating those bindings as hard uses turned every
    /// constant rewrite into a staging copy that the next pool binding straddled, which ended
    /// the render pass for each draw.
    void SoftenLastUsage() noexcept;

    /// True when the GPU produced bytes of the region that guest memory does not hold.
    [[nodiscard]] bool HasGpuProducedData(u64 offset, u64 size) const noexcept;

    /// The CPU rewrote or downloaded the range: guest memory holds its bytes again.
    void ClearProducedBits(u64 offset, u64 size) noexcept;

    /// Latest tick with uncompleted GPU writes to the region, zero when there are none.
    [[nodiscard]] u64 PendingWriteTick(u64 offset, u64 size) const noexcept;

    void ResetUsageTracking() noexcept;

    [[nodiscard]] u64 LastUsageTick() const noexcept {
        return last_usage_tick;
    }

private:
    static constexpr u64 USAGE_GRANULARITY_LOG2 = 10;
    /// Above this many pending ranges they are merged into one. Merging makes every range in
    /// between look written, which blocks deferred upload extensions, so the limit is generous:
    /// the lookups are rare (downloads, CPU reads) and the ranges of one tick merge as they
    /// touch.
    static constexpr size_t MAX_WRITE_RANGES = 512;
    /// Deferred uploads kept per buffer before new ones are copied right away. Games rewrite
    /// hundreds of constant buffer slots between two render target switches, lookups are
    /// binary searches so the list can be long.
    static constexpr size_t MAX_OVERLAYS = 512;

    struct BufferView {
        u32 offset;
        u32 size;
        VideoCore::Surface::PixelFormat format;
        id<MTLTexture> texture;
    };

    struct WriteRange {
        u64 begin;
        u64 end;
        u64 tick;
    };

public:
    /// A deferred upload. The runtime applies them all at once and bumps its generation, so an
    /// overlay of an older generation is already in the buffer and gets dropped. Overlays of a
    /// buffer never overlap each other and are sorted by offset.
    struct UploadOverlay {
        u64 offset;
        u64 size;
        u64 generation;
        id<MTLBuffer> staging;
        u64 staging_offset;
        /// Tick of shaders writing the staging copy through a written binding, zero when none.
        u64 gpu_write_tick = 0;
    };

    /// An upload copied by the side command buffer of tick: bindings use the buffer itself,
    /// the staging copy serves CPU reads and later rewrites of the range until the tick ends.
    /// Entries may overlap, the latest one wins.
    struct SideUpload {
        u64 offset;
        u64 size;
        u64 tick;
        id<MTLBuffer> staging;
        u64 staging_offset;
    };

private:
    static constexpr size_t MAX_SIDE_UPLOADS = 1024;

    /// Drops the side uploads of earlier ticks.
    void PruneSideUploads();

    /// The latest side upload holding every byte of [offset, size), null when there is none.
    [[nodiscard]] SideUpload* SideUploadCovering(u64 offset, u64 size) noexcept;

    /// True when a side upload of the current tick overlaps [offset, size).
    [[nodiscard]] bool HasSideUploadOverlapping(u64 offset, u64 size) const noexcept;

    /// Records the upload in the side command buffer. The region must not be used by the
    /// current tick and the copy must be blit aligned.
    void SideUploadBytes(u64 offset, std::span<const u8> data);

    /// Fills bytes with the latest contents of [offset, size) the CPU can see: guest memory,
    /// or the buffer where the GPU produced them. False when a GPU write is still pending on
    /// produced bytes.
    [[nodiscard]] bool ReadLatestBytes(u64 offset, u64 size, u8* bytes);

    /// Records a pending GPU write of the current tick without marking the region as used.
    void AddPendingWrite(u64 offset, u64 size) noexcept;

    /// Moves the usage bits of past ticks to previous_bits when the current tick changes.
    void RotateUsage(u64 tick) noexcept;

    /// Drops the overlays the runtime already applied.
    void PruneOverlays();

    /// Iterator to the first overlay ending after offset, the first candidate to overlap a
    /// range starting there.
    [[nodiscard]] std::vector<UploadOverlay>::iterator FirstOverlayEndingAfter(u64 offset) noexcept;

    /// The deferred upload holding every byte of [offset, size), null when there is none.
    [[nodiscard]] UploadOverlay* OverlayCovering(u64 offset, u64 size) noexcept;

    /// Grows the deferred uploads overlapping [offset, size) into one covering the whole range,
    /// filling the bytes between them from the buffer. Null when nothing overlaps or when it
    /// can't be done safely: the range is large, GPU writes are pending on the bytes in
    /// between, or shaders write one of the staging copies.
    [[nodiscard]] UploadOverlay* ExtendOverlay(u64 offset, u64 size);

    /// Replaces the staging copy of the deferred upload of the same range when one is still
    /// pending, so that only the latest data is copied at the end of the pass.
    void ReplacePendingUpload(u64 offset, u64 size, id<MTLBuffer> staging, u64 staging_offset);

    BufferCacheRuntime* runtime{};
    id<MTLBuffer> buffer;
    u8* mapped{};
    std::vector<BufferView> views;
    /// One bit per 1024 bytes. current_bits holds the regions used by current_usage_tick,
    /// previous_bits those used by older ticks up to previous_usage_tick. Splitting them lets
    /// regions become free once older work completes, even when the buffer is used every tick.
    std::vector<u64> current_bits;
    std::vector<u64> previous_bits;
    u64 current_usage_tick{};
    u64 previous_usage_tick{};
    u64 last_usage_tick{};
    std::vector<WriteRange> pending_writes;
    std::vector<UploadOverlay> overlays;
    std::vector<SideUpload> side_uploads;
    /// One bit per 1024 bytes: the GPU produced the bytes and guest memory lacks them. Set by
    /// produced writes, cleared when the CPU rewrites the range or downloads it.
    std::vector<u64> produced_bits;
    /// Granules of the current tick used by recorded commands that read exactly what they
    /// bind. Side uploads and storage buffer bindings are kept out of it, see SoftenLastUsage.
    std::vector<u64> hard_bits;
    /// Words and bits of hard_bits the last MarkUsage call newly set.
    std::vector<std::pair<u32, u64>> last_mark;
};

struct IndexBufferBinding {
    id<MTLBuffer> buffer;
    u64 offset = 0;
    /// Number of indices to draw. The first index has already been folded into offset, and
    /// generated quad indices hold absolute vertex numbers (draw with a base vertex of zero).
    u32 count = 0;
    MTLIndexType type = MTLIndexTypeUInt32;
    bool converted_quads = false;
};

class BufferCacheRuntime {
    friend Buffer;

    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

public:
    static constexpr u32 MAX_METAL_BUFFER_BINDINGS = 31;

    explicit BufferCacheRuntime(const Device& device, Scheduler& scheduler,
                                StagingBufferPool& staging_pool,
                                Tegra::MaxwellDeviceMemoryManager& device_memory);
    ~BufferCacheRuntime();

    void TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept;

    u64 CurrentTick() const noexcept;
    bool IsFree(u64 tick) const noexcept;
    void Wait(u64 tick);
    void Finish();

    u64 GetDeviceLocalMemory() const;
    u64 GetDeviceMemoryUsage() const;
    /// The buffer cache tracks its own allocations, see TextureCacheRuntime.
    bool CanReportMemoryUsage() const noexcept {
        return false;
    }

    u32 GetUniformBufferAlignment() const noexcept;
    u32 GetStorageBufferAlignment() const noexcept;

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size);
    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);
    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    bool CanReorderUpload(const Buffer&, std::span<const VideoCommon::BufferCopy>) const noexcept {
        return false;
    }

    void PreCopyBarrier();
    void PostCopyBarrier();

    /// Encodes the deferred buffer uploads. Ends the open render pass when there is one; the
    /// scheduler calls this when an encoder ends anyway.
    void ApplyPendingUploads();

    [[nodiscard]] u64 OverlayGeneration() const noexcept {
        return overlay_generation;
    }

    /// Logs how much unified memory Metal resources use.
    void ReportMemoryUsage() const;

    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool barrier);
    void CopyBuffer(Buffer& dst, id<MTLBuffer> src,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder_upload = false);
    void CopyBuffer(id<MTLBuffer> dst, Buffer& src,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier);

    void ClearBuffer(Buffer& dst, u32 offset, size_t size, u32 value);

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 first,
                         u32 count, Buffer& buffer, u32 offset, u32 size);
    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    void BindVertexBuffer(u32 index, id<MTLBuffer> buffer, u32 offset, u32 size, u32 stride);
    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);
    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    std::span<u8> BindMappedUniformBuffer(size_t stage, u32 binding_index, u32 size);
    void BindUniformBuffer(size_t stage, u32 binding_index, Buffer& buffer, u32 offset, u32 size);
    void BindComputeUniformBuffer(u32 binding_index, Buffer& buffer, u32 offset, u32 size);
    void BindStorageBuffer(size_t stage, u32 binding_index, Buffer& buffer, u32 offset, u32 size,
                           bool is_written);
    void BindComputeStorageBuffer(u32 binding_index, Buffer& buffer, u32 offset, u32 size,
                                  bool is_written);
    void BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                           VideoCore::Surface::PixelFormat format);
    void BindImageBuffer(Buffer& buffer, u32 offset, u32 size,
                         VideoCore::Surface::PixelFormat format);

    /// Binds the recorded vertex buffers starting at Metal buffer index first_index, skipping
    /// slots that already hold the same buffer and offset according to applied.
    void ApplyVertexBuffers(id<MTLRenderCommandEncoder> encoder, u32 first_index,
                            std::span<AppliedVertexBuffer> applied) const;

    /// Binds the uniform and storage buffers of a guest stage. Only the vertex (0) and fragment
    /// (4) stages have a native Metal equivalent.
    void ApplyStageBuffers(id<MTLRenderCommandEncoder> encoder, size_t stage, u32 uniform_base,
                           u32 storage_base) const;

    void ApplyComputeBuffers(id<MTLComputeCommandEncoder> encoder, u32 uniform_base,
                             u32 storage_base) const;

    [[nodiscard]] const IndexBufferBinding& IndexBinding() const noexcept {
        return index_binding;
    }

    /// Texture buffers in the order the buffer cache bound them since the last reset.
    [[nodiscard]] std::span<const id<MTLTexture>> TextureBuffers() const noexcept {
        return texture_buffers;
    }

    void ResetTextureBuffers() noexcept {
        texture_buffers.clear();
    }

    /// Recorded bindings, used to encode argument buffers. Out of range indices return an
    /// empty binding.
    [[nodiscard]] const BufferBinding& UniformBinding(size_t stage, u32 index) const noexcept {
        return stage < stages.size() && index < stages[stage].uniforms.size()
                   ? stages[stage].uniforms[index]
                   : empty_binding;
    }
    [[nodiscard]] const BufferBinding& StorageBinding(size_t stage, u32 index) const noexcept {
        return stage < stages.size() && index < stages[stage].storages.size()
                   ? stages[stage].storages[index]
                   : empty_binding;
    }
    [[nodiscard]] const BufferBinding& ComputeUniformBinding(u32 index) const noexcept {
        return index < compute_uniforms.size() ? compute_uniforms[index] : empty_binding;
    }
    [[nodiscard]] const BufferBinding& ComputeStorageBinding(u32 index) const noexcept {
        return index < compute_storages.size() ? compute_storages[index] : empty_binding;
    }

private:
    struct StageBindings {
        std::array<BufferBinding, VideoCommon::NUM_GRAPHICS_UNIFORM_BUFFERS> uniforms{};
        std::array<BufferBinding, VideoCommon::NUM_STORAGE_BUFFERS> storages{};
    };

    /// Copies between two shared buffers, on the GPU when Metal allows it.
    void EncodeCopy(id<MTLBuffer> src, u64 src_offset, id<MTLBuffer> dst, u64 dst_offset,
                    u64 size);

    /// Returns CPU readable bytes of a buffer region, waiting for pending GPU writes.
    const u8* ReadForCpu(Buffer& buffer, u64 offset, u64 size);

    void BindConvertedIndices(std::span<const u32> indices);

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;
    /// Guest memory, the source of the data the CPU writes to buffers.
    Tegra::MaxwellDeviceMemoryManager& device_memory;

    /// Deferred uploads of every buffer, only Metal handles so that buffers may move.
    struct PendingUpload {
        id<MTLBuffer> destination;
        u64 destination_offset;
        id<MTLBuffer> staging;
        u64 staging_offset;
        u64 size;
    };
    std::vector<PendingUpload> pending_uploads;
    u64 overlay_generation = 1;

    std::array<BufferBinding, VideoCommon::NUM_VERTEX_BUFFERS> vertex_buffers{};
    std::array<u64, VideoCommon::NUM_VERTEX_BUFFERS> vertex_strides{};
    u32 vertex_buffer_count = 0;
    std::array<BufferBinding, VideoCommon::NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback{};
    std::array<StageBindings, VideoCommon::NUM_STAGES> stages{};
    std::array<BufferBinding, VideoCommon::NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniforms{};
    std::array<BufferBinding, VideoCommon::NUM_STORAGE_BUFFERS> compute_storages{};
    std::vector<id<MTLTexture>> texture_buffers;
    BufferBinding empty_binding{};
    IndexBufferBinding index_binding{};
    std::vector<u32> index_scratch;
};

struct BufferCacheParams {
    using Runtime = Metal::BufferCacheRuntime;
    using Buffer = Metal::Buffer;
    using Async_Buffer = Metal::StagingBufferRef;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = true;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = true;
    static constexpr bool USE_MEMORY_MAPS = true;
    // Only image buffers can be written by shaders, texture buffers are not tracked as writes.
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = true;
    // Uploads write straight into the shared allocation through Buffer::ImmediateUpload.
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = false;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace Metal
