# uTLX attention on gfx1250

This experimental backend uses the uTLX extension to Triton for explicit TDM
loads and shared-memory synchronization. It requires an AMD gfx1250 GPU and a
compatible `utlx_plugin` build with `tdm_async_load` and `workgroup_barrier`.
Importing the SGLang kernel registry does not import the optional plugin.

## Interfaces

- `gfx1250.flash_attn`: dense, noncausal D128 MHA in FP16 or BF16, with an
  optional fixed-reference softmax and online-softmax recovery.
- `paged.unified_attention`: causal or noncausal D64/D128 attention with GQA
  ratios 1/2/4/8/16, ragged queries, page sizes 16/64/128, and a left window.
  Q and KV may use FP8 e4m3fn with scalar FP32 descales; output is FP16/BF16.
  It reads page tables and sequence lengths on the device during graph replay.

The paged implementation packs query heads sharing one KV head, overlaps page
loads with attention computation, and splits underfilled launches across KV
segments. Split counts scale with the device's CU count. The caller owns valid
page tables, monotonic query offsets and cache writes.
Long D64 GQA queries can load two pages per compute tile through indexed views
of the same shared allocation; this requires `local_reinterpret` support.

## SGLang integration

Select `--attention-backend tlx` to use the paged implementation for supported
prefill, decode and linear target-verification calls. The adapter inherits
AITER's cache writes, metadata construction and graph lifecycle. Calls outside
the kernel's contract, such as attention softcapping, delegate to AITER.

The backend is opt-in and requires the AITER runtime. It does not support MLA
or deterministic inference. It is not yet a DFlash draft-backend option; keep
`--speculative-draft-attention-backend triton` for DFlash2. A model monokernel
that bypasses the attention backend also bypasses this integration.

The kernels have numerical and forward-path coverage on MI450 DPX. Full Muse
Spark accuracy and AgentX performance validation are still required. Kernel
latencies vary by shape; this backend does not promise a gain for every prefill
or decode workload.

## Validation

With SGLang and the compatible uTLX extension installed, run on an idle gfx1250
GPU:

```bash
python -m pytest -q \
  test/manual/kernels/attention/test_tlx_dense_attention.py \
  test/manual/kernels/attention/test_tlx_paged_attention.py \
  test/manual/kernels/attention/test_tlx_backend.py
```

The tests use independent FP64 references, nonunit quantization scales,
shuffled page tables, cache-tail canaries, graph replays and the real SGLang
forward methods. Compiler coverage can run without a GPU:

```bash
python -m pytest -q test/manual/kernels/attention/test_tlx_codegen.py
```

These are manual tests because the repository does not currently provide a
gfx1250 GPU runner. They are not claimed as automated GPU CI coverage.
