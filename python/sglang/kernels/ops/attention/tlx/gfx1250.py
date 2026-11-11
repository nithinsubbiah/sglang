# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Ported from facebookexperimental/triton PR #4771 at
# 8f3a4cfd4a13e28dd42877e5c84cfc713a38a89f,
# third_party/tlx/ops/kernels/flash_attn/gfx1250.py.
# Requires the uTLX TDM operations and workgroup_barrier extension.

"""gfx1250 TDM/WMMA attention with device-side online-softmax recovery.

The fast path fixes the softmax reference at the first K tile and defers the
denominator reduction. Workgroups with unsafe normalization or accumulation
replay the attention using online softmax before the launch completes.
"""

import math

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _load_k(k_buf, slot, wait_count):
    tlx.tdm_async_wait(wait_count)
    return tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot)))


@triton.jit
def _load_v(v_buf, slot, wait_count):
    tlx.tdm_async_wait(wait_count)
    value = tlx.local_load(tlx.local_view(v_buf, slot))
    # All waves must finish reading the current K/V stages before any wave
    # can issue the next prefetch into their reusable LDS buffers.
    tlx.workgroup_barrier()
    return value


@triton.jit
def _compute_qk(
    q, k, cur_seq, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SEQLEN_K: tl.constexpr
):
    qk = tl.dot(q, k)
    qk_mask = (cur_seq + tl.arange(0, BLOCK_N))[None, :] < SEQLEN_K
    qk = tl.where(qk_mask, qk, float("-inf"))
    return qk


@triton.jit
def _compute_qk_no_mask(q, k):
    return tl.dot(q, k)


@triton.jit
def _softmax_part0(qk, m_i, scale_ln2):
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    q_shifted = (qk - m_ij[:, None]) * scale_ln2
    p = tl.math.exp2(q_shifted)
    alpha = tl.math.exp2((m_i - m_ij) * scale_ln2)
    return p, alpha, m_ij


@triton.jit
def _softmax_part1(
    p, l_i, acc, alpha, P_DTYPE: tl.constexpr, FIXED_REFERENCE: tl.constexpr
):
    if FIXED_REFERENCE:
        parts: tl.constexpr = l_i.shape[1]
        terms = tl.reshape(p, [p.shape[0], p.shape[1] // (parts * 8), parts, 8])
        terms = tl.sum(tl.sum(terms, 3), 1)
        l_i += terms
    else:
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
    p_low = p.to(P_DTYPE)
    return p_low, l_i, acc


@triton.jit
def _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE: tl.constexpr):
    if FIXED_REFERENCE:
        p = tl.math.exp2(scale_ln2 * qk - (m_i * scale_ln2)[:, None])
        alpha = tl.full(m_i.shape, 1.0, tl.float32)
        return p, alpha, m_i
    else:
        return _softmax_part0(qk, m_i, scale_ln2)


@triton.jit
def _needs_online_replay(
    acc, denominator, reference, scale_ln2, row_start, SEQLEN_Q: tl.constexpr
):
    # Each valid row has a unit contribution from the first tile. Reject lost
    # normalization or overflow, including overflow when P is converted to FP16.
    max_denominator: tl.constexpr = 1.0e30  # Leave FP32 headroom for normalization.
    max_reference_shift: tl.constexpr = 64.0  # Bound scale-minus-shift cancellation.
    fp32_max: tl.constexpr = 3.402823466e38
    bad_output = tl.sum((~(tl.abs(acc) <= fp32_max)).to(tl.int32), 1) > 0
    bad_row = ~((denominator > 0) & (denominator < max_denominator))
    bad_row |= (tl.abs(reference * scale_ln2) > max_reference_shift) | bad_output
    valid = row_start + tl.arange(0, acc.shape[0]) < SEQLEN_Q
    return tl.sum((valid & bad_row).to(tl.int32), 0) > 0


@triton.jit
def _attention_pass(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    SM_SCALE: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    FIXED_REFERENCE: tl.constexpr,
    QUERY_TILES_FIRST: tl.constexpr,
):
    NUM_BUFFERS: tl.constexpr = 2
    # Carry low-precision P across iterations to shorten FP32 live ranges.
    PREPARE_NEXT_P: tl.constexpr = FIXED_REFERENCE and BLOCK_M == 128 and BLOCK_N == 64
    # Overlap initial Q/K/V transfers where the extra LDS lifetime pays off.
    PREFETCH_Q: tl.constexpr = (
        HEAD_SZ == 128
        and (BLOCK_M == 32 or (BLOCK_M == 64 and SEQLEN_K < 512))
        and SEQLEN_K <= 768
    )
    scale_ln2: tl.constexpr = SM_SCALE * RCP_LN2

    off_z = tl.program_id(0).to(tl.int64)
    if QUERY_TILES_FIRST:
        query_blocks: tl.constexpr = tl.cdiv(SEQLEN_Q, BLOCK_M)
        tile = tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        off_h = tile // query_blocks
        off_m = (tile % query_blocks) * BLOCK_M
    else:
        off_h = tl.program_id(1)
        off_m = tl.program_id(2) * BLOCK_M

    off_h = off_h.to(tl.int64)
    q_desc = tlx.make_tensor_descriptor(
        None,
        q_ptr + off_z * stride_qz + off_h * stride_qh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_qm, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    # uTLX does not yet infer TDM-compatible layouts from descriptor users.
    # Pin the physical LDS image; its generic swizzled default is invalid here.
    q_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(HEAD_SZ, 8)], [BLOCK_M, HEAD_SZ], [1, 0]
    )
    v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(HEAD_SZ, 8)], [BLOCK_N, HEAD_SZ], [1, 0]
    )
    q_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(q_ptr), 1, layout=q_layout)
    tlx.tdm_async_load(q_desc, tlx.local_view(q_buf, 0), [off_m, 0])
    if not PREFETCH_Q:
        tlx.tdm_async_wait(0)
        q = tlx.local_load(tlx.local_view(q_buf, 0))

    k_desc = tlx.make_tensor_descriptor(
        None,
        k_ptr + off_z * stride_kz + off_h * stride_kh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_kn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    v_desc = tlx.make_tensor_descriptor(
        None,
        v_ptr + off_z * stride_vz + off_h * stride_vh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_vn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    o_desc = tlx.make_tensor_descriptor(
        None,
        o_ptr + off_z * stride_oz + off_h * stride_oh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_om, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    if HEAD_SZ == 128:
        # Pad K rows for the transposed WMMA operand loads.
        k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(HEAD_SZ, 8)], [BLOCK_N, HEAD_SZ], [1, 0]
        )
        k_buf = tlx.local_alloc(
            (BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS, layout=k_layout
        )
    else:
        k_buf = tlx.local_alloc(
            (BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS, layout=v_layout
        )
    v_buf = tlx.local_alloc(
        (BLOCK_N, HEAD_SZ), tlx.dtype_of(v_ptr), NUM_BUFFERS, layout=v_layout
    )
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(o_ptr), 1, layout=q_layout)

    ITERS_IN_PROLOGUE_EPILOGUE: tl.constexpr = 3
    n_blocks_n = max(
        (SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1
    )
    has_remainder: tl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N)
    if has_remainder:
        n_blocks_n = n_blocks_n - 1

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if FIXED_REFERENCE:
        # Wider query tiles benefit from packed updates to four partial sums.
        parts: tl.constexpr = min(4 if BLOCK_M >= 128 else 2, BLOCK_N // 8)
        l_i = tl.zeros([BLOCK_M, parts], dtype=tl.float32)
    else:
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SZ], dtype=tl.float32)

    if SEQLEN_K <= 2 * BLOCK_N:
        for key_start in tl.static_range(0, SEQLEN_K, BLOCK_N):
            tlx.tdm_async_load(k_desc, tlx.local_view(k_buf, 0), [key_start, 0])
            tlx.tdm_async_load(v_desc, tlx.local_view(v_buf, 0), [key_start, 0])
            if PREFETCH_Q and key_start == 0:
                tlx.tdm_async_wait(2)
                q = tlx.local_load(tlx.local_view(q_buf, 0))
            k = _load_k(k_buf, 0, 1)
            qk = _compute_qk(q, k, key_start, BLOCK_M, BLOCK_N, SEQLEN_K)
            if key_start == 0:
                p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
            else:
                p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
            v = _load_v(v_buf, 0, 0)
            acc = tl.dot(p, v, acc)
    else:
        block_min = 0
        block_max = n_blocks_n * BLOCK_N

        tlx.tdm_async_load(k_desc, tlx.local_view(k_buf, 0), [0, 0])
        tlx.tdm_async_load(k_desc, tlx.local_view(k_buf, 1), [BLOCK_N, 0])
        tlx.tdm_async_load(v_desc, tlx.local_view(v_buf, 0), [0, 0])

        if PREFETCH_Q:
            tlx.tdm_async_wait(3)
            q = tlx.local_load(tlx.local_view(q_buf, 0))
        k = _load_k(k_buf, 0, 2)
        qk = _compute_qk(q, k, 0, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

        tlx.tdm_async_load(k_desc, tlx.local_view(k_buf, 0), [2 * BLOCK_N, 0])
        tlx.tdm_async_load(v_desc, tlx.local_view(v_buf, 1), [BLOCK_N, 0])

        k = _load_k(k_buf, 1, 3)

        if PREPARE_NEXT_P:
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
        iter_id = 0
        unroll: tl.constexpr = 2 if not FIXED_REFERENCE and SEQLEN_K < 512 else 1
        for block_id in tl.range(
            block_min, block_max, BLOCK_N, loop_unroll_factor=unroll
        ):
            t_2 = block_id + 2 * BLOCK_N
            t_3 = block_id + 3 * BLOCK_N

            qk = _compute_qk_no_mask(q, k)
            if not PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(
                    p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE
                )
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.tdm_async_load(
                k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0]
            )
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            if PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.tdm_async_load(
                v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0]
            )
            iter_id += 1

        if has_remainder:
            t_1 = iter_id * BLOCK_N + BLOCK_N
            t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
            t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

            qk = _compute_qk(q, k, t_1, BLOCK_M, BLOCK_N, SEQLEN_K)
            if not PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(
                    p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE
                )
            v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
            tlx.tdm_async_load(
                k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0]
            )
            acc = tl.dot(p, v, acc)
            p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
            if PREPARE_NEXT_P:
                p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, True)
            k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
            tlx.tdm_async_load(
                v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0]
            )
            iter_id += 1

        epilogue_offset = (iter_id - 1) * BLOCK_N
        t_2 = epilogue_offset + 2 * BLOCK_N
        t_3 = epilogue_offset + 3 * BLOCK_N

        if not PREPARE_NEXT_P:
            p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        acc = tl.dot(p, v, acc)

        qk = _compute_qk(q, k, t_2, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)

        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 1)
        tlx.tdm_async_load(
            v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_3, 0]
        )

        qk = _compute_qk(q, k, t_3, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, (iter_id + 1) % NUM_BUFFERS, 1)
        acc = tl.dot(p, v, acc)

        p, alpha, m_i = _softmax_later(qk, m_i, scale_ln2, FIXED_REFERENCE)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha, q.dtype, FIXED_REFERENCE)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 0)
        acc = tl.dot(p, v, acc)

    if FIXED_REFERENCE:
        l_i = tl.sum(l_i, 1)
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    o_view = tlx.local_view(o_buf, 0)
    output = acc
    if o_ptr.dtype.element_ty == tl.float16:
        # INVARIANT: convex combinations of finite FP16 values cannot exceed the FP16 range.
        output = tl.where(
            tl.abs(output) < float("inf"),
            tl.minimum(tl.maximum(output, -65504.0), 65504.0),
            output,
        )
    tlx.local_store(o_view, output.to(o_ptr.dtype.element_ty))
    tlx.tdm_async_store(o_desc, o_view, [off_m, 0])
    tlx.tdm_async_wait(0)

    if FIXED_REFERENCE:
        # The completed store above is provisional: recovery overwrites it.
        return _needs_online_replay(acc, l_i, m_i, scale_ln2, off_m, SEQLEN_Q)
    return False


# Keep uncommon recovery out of the fast loop's register schedule.
@triton.jit(noinline=True)
def _online_replay(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    SM_SCALE: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    QUERY_TILES_FIRST: tl.constexpr,
):
    _attention_pass(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        SM_SCALE,
        SEQLEN_Q,
        SEQLEN_K,
        BLOCK_M,
        BLOCK_N,
        HEAD_SZ,
        False,
        QUERY_TILES_FIRST,
    )


@triton.jit(do_not_specialize=["BATCH", "NUM_HEADS"])
def attn_fwd_tdm_pipelined_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    SM_SCALE: tl.constexpr,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    USE_FIXED_REFERENCE: tl.constexpr = False,
    QUERY_TILES_FIRST: tl.constexpr = False,
    BATCH=1,
    NUM_HEADS=1,
):
    if USE_FIXED_REFERENCE:
        retry = _attention_pass(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_on,
            SM_SCALE,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            True,
            QUERY_TILES_FIRST,
        )
        if retry:
            _online_replay(
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                stride_oz,
                stride_oh,
                stride_om,
                stride_on,
                SM_SCALE,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_M,
                BLOCK_N,
                HEAD_SZ,
                QUERY_TILES_FIRST,
            )
    else:
        _attention_pass(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_on,
            SM_SCALE,
            SEQLEN_Q,
            SEQLEN_K,
            BLOCK_M,
            BLOCK_N,
            HEAD_SZ,
            False,
            QUERY_TILES_FIRST,
        )


def _select_tile_sizes(batch, heads, seqlen_q, seqlen_k, head_size, compute_units):
    if head_size != 128:
        return 128, 128
    query_blocks = batch * heads * triton.cdiv(seqlen_q, 128)
    if query_blocks < compute_units:
        narrow_blocks = batch * heads * triton.cdiv(seqlen_q, 64)
        if seqlen_k > 1024:
            return (32 if narrow_blocks <= compute_units // 2 else 64), 128
        if seqlen_k >= 512 and narrow_blocks < compute_units:
            if seqlen_k <= 768 or narrow_blocks <= compute_units // 8:
                return 32, 128
        return 64, 128
    if seqlen_k < 512:
        return 128, 128
    wide = seqlen_k >= 1024 and batch * heads * (seqlen_q // 256) >= compute_units
    return (256 if wide else 128), 64


def _use_query_tiles_first(heads, seqlen_k, head_size):
    return head_size == 128 and seqlen_k >= 65536 and heads >= 32


def _validate_inputs(q, k, v, causal, sm_scale):
    if causal:
        raise ValueError(
            "uTLX dense attention currently supports noncausal attention only"
        )
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("uTLX dense attention expects rank-4 BHSD tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v)
    ):
        raise ValueError("uTLX dense attention expects matching FP16 or BF16 dtypes")
    if (
        q.shape[-1] != 128
        or k.shape[:2] != q.shape[:2]
        or k.shape[-1] != 128
        or v.shape != k.shape
    ):
        raise ValueError(
            "uTLX dense attention expects D128 MHA and matching K/V shapes"
        )
    if any(min(t.shape) <= 0 for t in (q, k, v)):
        raise ValueError("uTLX dense attention expects nonempty tensors")
    if any(t.stride(-1) != 1 or min(t.stride()) <= 0 for t in (q, k, v)):
        raise ValueError(
            "uTLX dense attention requires positive strides and contiguous heads"
        )
    if (
        not isinstance(sm_scale, (int, float))
        or not math.isfinite(sm_scale)
        or sm_scale <= 0
    ):
        raise ValueError(
            "uTLX dense attention requires a positive finite softmax scale"
        )
    if not q.is_cuda or any(t.device != q.device for t in (k, v)):
        raise ValueError("uTLX dense attention expects Q/K/V on one GPU")
    if torch.version.hip is None or not torch.cuda.get_device_properties(
        q.device
    ).gcnArchName.startswith("gfx1250"):
        raise ValueError("uTLX dense attention requires gfx1250")
    if any(
        t.data_ptr() % 16
        or any(stride * t.element_size() % 16 for stride in t.stride()[:-1])
        for t in (q, k, v)
    ):
        raise ValueError(
            "uTLX dense attention requires 16-byte aligned bases and row strides"
        )


def flash_attn(q, k, v, causal=False, sm_scale=None, *, use_fixed_reference=False):
    """Dense noncausal D128 MHA on gfx1250, preserving input shape and dtype.

    Inputs use BHSD layout with a contiguous head dimension. The optional uTLX
    plugin must provide TDM operations and ``workgroup_barrier``. Stable online
    softmax is the default while the fixed-reference recovery path is validated
    separately. This kernel is not yet a registered SGLang attention backend.
    """
    if sm_scale is None and q.ndim == 4:
        sm_scale = q.shape[-1] ** -0.5 if q.shape[-1] else None
    _validate_inputs(q, k, v, causal, sm_scale)
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    units = torch.cuda.get_device_properties(q.device).multi_processor_count
    block_m, block_n = _select_tile_sizes(batch, heads, sq, sk, dim, units)
    output = torch.empty_like(q)
    grid = (batch, heads, triton.cdiv(sq, block_m))
    attn_fwd_tdm_pipelined_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        sm_scale,
        sq,
        sk,
        block_m,
        block_n,
        dim,
        USE_FIXED_REFERENCE=use_fixed_reference and sk >= 512,
        QUERY_TILES_FIRST=_use_query_tiles_first(heads, sk, dim),
        BATCH=batch,
        NUM_HEADS=heads,
        num_warps=2 if block_m == 32 else 4,
        waves_per_eu=1,
    )
    return output
