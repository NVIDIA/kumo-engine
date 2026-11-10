/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Benchmark-only BCC source. Field offsets are validated against host BTF. */
#include <uapi/linux/ptrace.h>

struct request_state {
    u64 bytes;
    u32 operation;
    u32 issue_mask;
    u32 requeued;
    u32 root_issued;
};
struct counters {
    u64 reads;
    u64 read_bytes;
    u64 writes;
    u64 write_bytes;
    u64 carried_in_reads;
    u64 carried_out_reads;
    u64 root_issued_reads;
    u64 root_issued_writes;
    u64 issued;
    u64 merged_away;
    u64 requeues;
    u64 reissues;
    u64 partials;
};
struct trace_state {
    u64 pending_reads;
    u64 pending_writes;
    u64 mask;
    u64 marker_count;
    u32 marker_pid;
};
struct marker {
    u64 before_ns;
    u64 after_ns;
    u64 pending_reads;
    u64 pending_writes;
    u32 pid;
    u32 phase;
    u32 entering;
    u32 mask;
};
BPF_HASH(requests, u64, struct request_state, REQUEST_CAPACITY);
BPF_PERCPU_ARRAY(counts, struct counters, 7);
BPF_PERCPU_ARRAY(diagnostics, u64, 11);
BPF_ARRAY(state, struct trace_state, 1);
BPF_ARRAY(markers, struct marker, 16);

/* Diagnostics0..9 are fatal;10 counts untracked device completions, mostly
 * foreign requests. Target insert/probe failures are separately fatal. */
static __always_inline void diagnostic(u32 key) {
    u64 *value = diagnostics.lookup(&key);
    if (value) __sync_fetch_and_add(value, 1);
}
static __always_inline int read64(u64 pointer, u32 offset, u64 *value) {
    if (!pointer || bpf_probe_read_kernel(value, sizeof(*value),
                                        (void *)(pointer + offset))) {
        diagnostic(0);
        return -1;
    }
    return 0;
}
static __always_inline int read32(u64 pointer, u32 offset, u32 *value) {
    if (!pointer || bpf_probe_read_kernel(value, sizeof(*value),
                                        (void *)(pointer + offset))) {
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
    if (*operation > 1) return 0;
    if (read64(request, OFF_REQUEST_Q, &queue) ||
        read64(queue, OFF_REQUEST_QUEUE_DISK, &disk)) return 0;
    if (!disk) return 0;
    if (read32(disk, OFF_GENDISK_MAJOR, &major) ||
        read32(disk, OFF_GENDISK_FIRST_MINOR, &minor)) return 0;
    return major == DEVICE_MAJOR && minor == DEVICE_MINOR;
}
static __always_inline int owner(u64 request, u64 *cgid, u32 *root_issued) {
    u64 bio = 0, blkg = 0, blkcg = 0, cgroup = 0, kn = 0;
    u32 flags = 0;
    if (read64(request, OFF_REQUEST_BIO, &bio)) return -1;
    if (!bio) { diagnostic(1); return -1; }
    if (read64(bio, OFF_BIO_BI_BLKG, &blkg) ||
        read64(blkg, OFF_BLKCG_GQ_BLKCG, &blkcg) ||
        read64(blkcg, OFF_BLKCG_CSS + OFF_CGROUP_SUBSYS_STATE_CGROUP, &cgroup) ||
        read64(cgroup, OFF_CGROUP_KN, &kn) ||
        read64(kn, OFF_KERNFS_NODE_ID, cgid) ||
        read32(bio, OFF_BIO_BI_OPF, &flags)) return -1;
    *root_issued = (flags & ROOT_ISSUE_FLAGS) != 0;
    return 0;
}
static __always_inline struct trace_state *current_state(void) {
    u32 zero = 0;
    return state.lookup(&zero);
}
static __always_inline void decrement(struct trace_state *s, u32 operation) {
    if (operation == 0) __sync_fetch_and_add(&s->pending_reads, -1);
    else __sync_fetch_and_add(&s->pending_writes, -1);
}

RAW_TRACEPOINT_PROBE(block_rq_issue) {
    u64 request = ctx->args[0], cgid = 0;
    u32 operation = 0, root_issued = 0, zero = 0;
    if (!device_request(request, &operation)) return 0;
    if (owner(request, &cgid, &root_issued) || cgid != TARGET_CGROUP) return 0;
    struct trace_state *s = current_state();
    struct counters *total = counts.lookup(&zero);
    if (!s || !total) return 0;
    struct request_state *old = requests.lookup(&request);
    if (old) {
        if (!old->requeued || old->operation != operation) diagnostic(3);
        old->requeued = 0;
        __sync_fetch_and_add(&total->reissues, 1);
        return 0;
    }
    struct request_state value = {};
    value.operation = operation;
    value.issue_mask = s->mask;
    value.root_issued = root_issued;
    if (requests.insert(&request, &value)) { diagnostic(2); return 0; }
    __sync_fetch_and_add(&total->issued, 1);
    if (operation == 0) __sync_fetch_and_add(&s->pending_reads, 1);
    else __sync_fetch_and_add(&s->pending_writes, 1);
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_complete) {
    u64 request = ctx->args[0];
    struct request_state *value = requests.lookup(&request);
    if (!value) return 0;
    __sync_fetch_and_add(&value->bytes, ctx->args[2]);
    u32 remaining = 0, zero = 0;
    if (!read32(request, OFF_REQUEST_DATA_LEN, &remaining) &&
        ctx->args[2] < remaining) {
        struct counters *total = counts.lookup(&zero);
        if (total) __sync_fetch_and_add(&total->partials, 1);
    }
    if (ctx->args[1]) diagnostic(4);
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_requeue) {
    u64 request = ctx->args[0];
    struct request_state *value = requests.lookup(&request);
    if (value) {
        u32 zero = 0;
        struct counters *total = counts.lookup(&zero);
        if (total) __sync_fetch_and_add(&total->requeues, 1);
        value->requeued = 1;
    }
    return 0;
}
RAW_TRACEPOINT_PROBE(block_rq_merge) {
    u64 request = ctx->args[0];
    struct request_state *value = requests.lookup(&request);
    if (value) {
        u32 zero = 0;
        struct counters *total = counts.lookup(&zero);
        struct trace_state *s = current_state();
        if (value->bytes) diagnostic(7);
        if (total) __sync_fetch_and_add(&total->merged_away, 1);
        if (s) decrement(s, value->operation);
        if (requests.delete(&request)) diagnostic(9);
    }
    return 0;
}
static __always_inline void complete(struct counters *count,
                                     struct request_state *request) {
    if (request->operation == 0) {
        __sync_fetch_and_add(&count->reads, 1);
        __sync_fetch_and_add(&count->read_bytes, request->bytes);
        if (request->root_issued) __sync_fetch_and_add(&count->root_issued_reads, 1);
    } else {
        __sync_fetch_and_add(&count->writes, 1);
        __sync_fetch_and_add(&count->write_bytes, request->bytes);
        if (request->root_issued) __sync_fetch_and_add(&count->root_issued_writes, 1);
    }
}
RAW_TRACEPOINT_PROBE(block_io_done) {
    u64 request = ctx->args[0];
    struct request_state *value = requests.lookup(&request);
    if (!value) {
        u32 operation = 0;
        if (device_request(request, &operation)) diagnostic(10);
        return 0;
    }
    struct trace_state *s = current_state();
    if (!s) return 0;
    u32 zero = 0, mask = s->mask;
    struct counters *total = counts.lookup(&zero);
    if (total) complete(total, value);
#pragma unroll
    for (u32 phase = 1; phase < 7; phase++) {
        struct counters *count = counts.lookup(&phase);
        if (!count) continue;
        if (mask & (1U << phase)) {
            complete(count, value);
            if (value->operation == 0 && !(value->issue_mask & (1U << phase)))
                __sync_fetch_and_add(&count->carried_in_reads, 1);
        } else if (value->operation == 0 && (value->issue_mask & (1U << phase))) {
            __sync_fetch_and_add(&count->carried_out_reads, 1);
        }
    }
    decrement(s, value->operation);
    if (requests.delete(&request)) diagnostic(9);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_prctl) {
    if (args->option != MARKER_MAGIC ||
        bpf_get_current_cgroup_id() != TARGET_CGROUP) return 0;
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct trace_state *s = current_state();
    if (!s) return 0;
    if (!s->marker_pid) s->marker_pid = pid;
    if (s->marker_pid != pid) { diagnostic(8); return 0; }
    u32 phase = args->arg2, entering = args->arg3;
    if (phase < 1 || phase > 6 || entering > 1) { diagnostic(5); return 0; }
    u32 index = __sync_fetch_and_add(&s->marker_count, 1);
    if (index >= 16) { diagnostic(6); return 0; }
    struct marker *mark = markers.lookup(&index);
    if (!mark) return 0;
    mark->before_ns = bpf_ktime_get_ns();
    u32 bit = 1U << phase;
    if ((entering && (s->mask & bit)) || (!entering && !(s->mask & bit))) diagnostic(5);
    if (entering) __sync_fetch_and_or(&s->mask, bit);
    else __sync_fetch_and_and(&s->mask, ~bit);
    mark->after_ns = bpf_ktime_get_ns();
    mark->pending_reads = s->pending_reads;
    mark->pending_writes = s->pending_writes;
    mark->pid = pid;
    mark->phase = phase;
    mark->entering = entering;
    mark->mask = s->mask;
    return 0;
}
