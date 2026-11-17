/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Diagnostic events only. All measured-device operations are tracked. */
#include <uapi/linux/ptrace.h>

struct request_state {
    u64 generation;
    u64 cgroup;
    u64 bytes;
    u32 operation;
    u32 active;
    u32 attribution_flags;
};
struct flush_token {
    struct request_state request;
    u64 start_time;
};
struct event {
    u64 timestamp;
    u64 request;
    u64 generation;
    u64 cgroup;
    u64 bytes;
    u32 kind;
    u32 operation;
    u32 phase;
    u32 entering;
    u32 pid;
    u32 cpu;
    u64 sequence;
};
struct trace_state {
    u64 pending;
    u64 bootstrap_ns;
    u64 end_ns;
    u64 capture_enabled;
    u64 pending_flush;
};
BPF_HASH(requests, u64, struct request_state, REQUEST_CAPACITY);
BPF_HASH(flush_tokens, u64, struct flush_token, REQUEST_CAPACITY);
BPF_STACK_TRACE(unknown_stacks, 128);
BPF_PERCPU_ARRAY(sequence, u64, 1);
BPF_PERCPU_ARRAY(diagnostics, u64, 10);
BPF_ARRAY(state, struct trace_state, 1);
BPF_PERF_OUTPUT(events);

static __always_inline int enabled(void) {
    u32 zero = 0;
    struct trace_state *s = state.lookup(&zero);
    return s && s->capture_enabled && !s->end_ns;
}

static __always_inline void diagnostic(u32 key) {
    u64 *value = diagnostics.lookup(&key);
    if (value) __sync_fetch_and_add(value, 1);
}
static __always_inline int read64(u64 pointer, u32 offset, u64 *value) {
    if (!pointer || bpf_probe_read_kernel(value, 8, (void *)(pointer + offset))) {
        diagnostic(0);
        return -1;
    }
    return 0;
}
static __always_inline int read32(u64 pointer, u32 offset, u32 *value) {
    if (!pointer || bpf_probe_read_kernel(value, 4, (void *)(pointer + offset))) {
        diagnostic(0);
        return -1;
    }
    return 0;
}
static __always_inline int device_request(u64 request, u32 *operation) {
    u64 queue = 0, disk = 0;
    u32 major = 0, minor = 0, flags = 0;
    if (read32(request, OFF_REQUEST_CMD_FLAGS, &flags)) return 0;
    *operation = flags & 255;
    if (read64(request, OFF_REQUEST_Q, &queue) ||
        read64(queue, OFF_REQUEST_QUEUE_DISK, &disk)) return 0;
    if (!disk) return 0;
    if (read32(disk, OFF_GENDISK_MAJOR, &major) ||
        read32(disk, OFF_GENDISK_FIRST_MINOR, &minor)) return 0;
    return major == DEVICE_MAJOR && minor == DEVICE_MINOR;
}
static __always_inline u64 bio_owner(u64 bio) {
    u64 blkg = 0, blkcg = 0, cgroup = 0, kn = 0, cgid = 0;
    if (read64(bio, OFF_BIO_BI_BLKG, &blkg) ||
        read64(blkg, OFF_BLKCG_GQ_BLKCG, &blkcg) ||
        read64(blkcg, OFF_BLKCG_CSS + OFF_CGROUP_SUBSYS_STATE_CGROUP, &cgroup) ||
        read64(cgroup, OFF_CGROUP_KN, &kn) ||
        read64(kn, OFF_KERNFS_NODE_ID, &cgid)) return 0;
    return cgid;
}
static __always_inline u64 owner(u64 request, u32 operation, u32 *flags) {
    u64 bio = 0, cgid = 0;
    if (read64(request, OFF_REQUEST_BIO, &bio)) return 0;
    /* Flush/driver commands need not have a BIO. They still occupy device depth. */
    if (!bio) {
        if (operation <= 1) diagnostic(1);
        *flags |= 4;
        return 0;
    }
    cgid = bio_owner(bio);
    if (cgid == TARGET_CGROUP) *flags |= 8;
#pragma unroll
    for (int index = 0; index < 8; index++) {
        if (!bio) break;
        if (index) {
            u64 member = bio_owner(bio);
            if (member != cgid) *flags |= 1;
            if (member == TARGET_CGROUP) *flags |= 8;
        }
        u64 next = 0;
        if (read64(bio, OFF_BIO_NEXT, &next)) { *flags |= 4; break; }
        bio = next;
    }
    if (bio) *flags |= 2;
    return cgid;
}
static __always_inline void emit(void *ctx, u32 kind, u64 pointer,
                                 struct request_state *request, u32 phase,
                                 u32 entering, u64 timestamp) {
    u32 zero = 0;
    u64 *seq = sequence.lookup(&zero);
    if (!seq) { diagnostic(7); return; }
    struct event value = {};
    value.timestamp = timestamp;
    value.request = pointer;
    if (request) {
        value.generation = request->generation;
        value.cgroup = request->cgroup;
        value.bytes = request->bytes;
        value.operation = request->operation;
        value.phase = request->attribution_flags;
    }
    value.kind = kind;
    if (!request) value.phase = phase;
    value.entering = entering;
    value.pid = bpf_get_current_pid_tgid() >> 32;
    value.cpu = bpf_get_smp_processor_id();
    value.sequence = __sync_fetch_and_add(seq, 1) + 1;
    if (events.perf_submit(ctx, &value, sizeof(value))) diagnostic(6);
}

static __always_inline void emit_unknown(void *ctx, u64 pointer, u32 operation) {
    u32 cmd_flags = 0, rq_flags = 0, remaining = 0;
    struct request_state metadata = {};
    if (read32(pointer, OFF_REQUEST_CMD_FLAGS, &cmd_flags) ||
        read32(pointer, OFF_REQUEST_RQ_FLAGS, &rq_flags) ||
        read32(pointer, OFF_REQUEST_DATA_LEN, &remaining) ||
        read64(pointer, OFF_REQUEST_START_TIME_NS, &metadata.generation)) return;
    /* UNKNOWN encoding v1: these are diagnostic fields, never ownership,
     * completed bytes, or a proved request lifecycle. Preserve negative stack
     * errors as signed-u32 bits; only this rejected branch collects stacks. */
    metadata.cgroup = cmd_flags;
    metadata.bytes = remaining;
    metadata.operation = operation;
    metadata.attribution_flags = rq_flags;
    int stack = unknown_stacks.get_stackid(ctx, 0);
    emit(ctx, 8, pointer, &metadata, 0, (u32)stack, bpf_ktime_get_ns());
}

RAW_TRACEPOINT_PROBE(block_rq_issue) {
    if (!enabled()) return 0;
    u64 pointer = ctx->args[0];
    u32 operation = 0, zero = 0;
    if (!device_request(pointer, &operation)) return 0;
    if (flush_tokens.lookup(&pointer)) { diagnostic(8); return 0; }
    u64 now = bpf_ktime_get_ns();
    struct request_state *old = requests.lookup(&pointer);
    if (old) {
        if (old->active || old->operation != operation) diagnostic(3);
        u32 flags = 0;
        u64 cgroup = owner(pointer, operation, &flags);
        /* A changed BIO chain cannot silently retain earlier pure-target
         * attribution. This diagnostic rejects such a requeue/merge trial. */
        if (cgroup != old->cgroup || flags != old->attribution_flags) diagnostic(3);
        old->active = 1;
        emit(ctx, 4, pointer, old, 0, 0, now);
        return 0;
    }
    struct request_state value = {};
    value.generation = now;
    value.cgroup = owner(pointer, operation, &value.attribution_flags);
    value.operation = operation;
    value.active = 1;
    if (requests.insert(&pointer, &value)) { diagnostic(2); return 0; }
    struct trace_state *s = state.lookup(&zero);
    if (s) __sync_fetch_and_add(&s->pending, 1);
    emit(ctx, 1, pointer, &value, 0, 0, now);
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_complete) {
    if (!enabled()) return 0;
    u64 pointer = ctx->args[0];
    struct request_state *value = requests.lookup(&pointer);
    if (!value) return 0;
    __sync_fetch_and_add(&value->bytes, ctx->args[2]);
    if (ctx->args[1]) diagnostic(4);
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_requeue) {
    if (!enabled()) return 0;
    u64 pointer = ctx->args[0];
    struct request_state *value = requests.lookup(&pointer);
    if (value) {
        if (!value->active) diagnostic(3);
        value->active = 0;
        emit(ctx, 3, pointer, value, 0, 0, bpf_ktime_get_ns());
    }
    return 0;
}
static __always_inline int remove_request(void *ctx, u64 pointer, u32 kind) {
    u32 zero = 0, operation = 0;
    struct request_state *value = requests.lookup(&pointer);
    if (!value) {
        struct flush_token *token = flush_tokens.lookup(&pointer);
        if (token && kind != 2) { diagnostic(8); return 0; }
        if (token && kind == 2) {
            u32 flags = 0, remaining = 0;
            u64 start_time = 0;
            if (!device_request(pointer, &operation) ||
                read32(pointer, OFF_REQUEST_RQ_FLAGS, &flags) ||
                read32(pointer, OFF_REQUEST_DATA_LEN, &remaining) ||
                read64(pointer, OFF_REQUEST_START_TIME_NS, &start_time)) return 0;
            if (operation != 1 || flags & RQF_FLUSH_SEQ_MASK || remaining ||
                !start_time || start_time != token->start_time) {
                diagnostic(8);
                return 0;
            }
            /* blk_flush_complete_seq restores the already-completed DATA
             * request and calls blk_mq_end_request again. This is logical
             * completion, without a second hardware dispatch or payload. */
            struct request_state metadata = token->request;
            metadata.bytes = start_time;
            emit(ctx, 10, pointer, &metadata, 0, flags, bpf_ktime_get_ns());
            struct trace_state *s = state.lookup(&zero);
            if (s) __sync_fetch_and_add(&s->pending_flush, -1);
            if (flush_tokens.delete(&pointer)) diagnostic(9);
            return 0;
        }
        /* A request merged before its first issue never occupied dispatched
         * depth and need not have an entry in our issue-started lifecycle. */
        if (kind != 5 && device_request(pointer, &operation))
            emit_unknown(ctx, pointer, operation);
        return 0;
    }
    if (kind == 5 && value->bytes) diagnostic(4);
    if (kind == 2 && value->operation == 1) {
        u32 flags = 0, remaining = 0;
        if (read32(pointer, OFF_REQUEST_RQ_FLAGS, &flags)) return 0;
        if (flags & RQF_FLUSH_SEQ_MASK) {
            struct flush_token token = {};
            token.request = *value;
            if (!value->active || !value->bytes ||
                read32(pointer, OFF_REQUEST_DATA_LEN, &remaining) ||
                read64(pointer, OFF_REQUEST_START_TIME_NS, &token.start_time)) {
                diagnostic(8);
                return 0;
            }
            if (remaining || !token.start_time || token.start_time > value->generation) {
                diagnostic(8);
                return 0;
            }
            if (flush_tokens.insert(&pointer, &token)) { diagnostic(2); return 0; }
            struct trace_state *s = state.lookup(&zero);
            if (s) __sync_fetch_and_add(&s->pending_flush, 1);
            struct request_state metadata = *value;
            metadata.bytes = token.start_time;
            emit(ctx, 11, pointer, &metadata, 0, flags, bpf_ktime_get_ns());
        }
    }
    emit(ctx, kind, pointer, value, 0, 0, bpf_ktime_get_ns());
    struct trace_state *s = state.lookup(&zero);
    if (s) __sync_fetch_and_add(&s->pending, -1);
    if (requests.delete(&pointer)) diagnostic(5);
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_merge) {
    if (!enabled()) return 0;
    return remove_request(ctx, ctx->args[0], 5);
}
RAW_TRACEPOINT_PROBE(block_io_done) {
    if (!enabled()) return 0;
    return remove_request(ctx, ctx->args[0], 2);
}
TRACEPOINT_PROBE(syscalls, sys_enter_prctl) {
    u32 pid = bpf_get_current_pid_tgid() >> 32, zero = 0;
    struct trace_state *s = state.lookup(&zero);
    if (!s || !s->capture_enabled) return 0;
    if (args->option == ANCHOR_MAGIC && pid == OBSERVER_PID) {
        if (!s->pending && !s->pending_flush && !s->bootstrap_ns && args->arg2 == 0) {
            s->bootstrap_ns = bpf_ktime_get_ns();
            emit(args, 7, 0, 0, 0, 0, s->bootstrap_ns);
        } else if (!s->pending && !s->pending_flush && s->bootstrap_ns && !s->end_ns && args->arg2 == 1) {
            s->end_ns = bpf_ktime_get_ns();
            emit(args, 9, 0, 0, 0, 0, s->end_ns);
        }
        return 0;
    }
    if (!s->end_ns && args->option == MARKER_MAGIC && bpf_get_current_cgroup_id() == TARGET_CGROUP)
        emit(args, 6, 0, 0, args->arg2, args->arg3, bpf_ktime_get_ns());
    return 0;
}
