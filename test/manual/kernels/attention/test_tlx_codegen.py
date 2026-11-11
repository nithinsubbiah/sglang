"""Compile the actual SGLang uTLX kernel without launching it on a GPU."""

import unittest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from sglang.kernels.ops.attention.tlx.gfx1250 import attn_fwd_tdm_pipelined_kernel
from sglang.test.test_utils import CustomTestCase


@triton.jit
def _plain_add(x, y):
    offsets = tl.arange(0, 128)
    tl.store(y + offsets, tl.load(x + offsets) + 1)


@gluon.jit
def _gluon_add(x, y):
    offsets = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(y + offsets, gl.load(x + offsets) + 1)


def _check_other_frontend(frontend):
    source = (
        ASTSource(_plain_add, signature={"x": "*fp32", "y": "*fp32"})
        if frontend == "triton"
        else GluonASTSource(_gluon_add, signature={"x": "*fp32", "y": "*fp32"})
    )
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1250", 32))
    assert compiled.asm["hsaco"]


def _check_dense_codegen(dtype, sq, sk, block_m, block_n, warps, fixed):
    fn = attn_fwd_tdm_pipelined_kernel
    constants = {
        "SM_SCALE": 128**-0.5,
        "SEQLEN_Q": sq,
        "SEQLEN_K": sk,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "HEAD_SZ": 128,
        "USE_FIXED_REFERENCE": fixed,
        "QUERY_TILES_FIRST": False,
    }
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            signature[name] = "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = "*" + dtype
        elif name in ("stride_qk", "stride_kk", "stride_vk", "stride_on"):
            constants[name] = 1
            signature[name] = "constexpr"
        else:
            signature[name] = "i64" if name.startswith("stride_") else "i32"
    compiled = triton.compile(
        ASTSource(fn, signature=signature, constexprs=constants),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": warps, "waves_per_eu": 1},
    )
    assert "amdg.async_tdm_copy_global_to_local" in compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_local_to_global" in compiled.asm["ttgir"]
    assert "rocdl.sched.barrier none" in compiled.asm["ttgir"]
    assert "wmma" in compiled.asm["amdgcn"]


class TestTlxCodegen(CustomTestCase):
    def test_other_frontends_still_compile(self):
        for frontend in ("triton", "gluon"):
            with self.subTest(frontend=frontend):
                _check_other_frontend(frontend)

    def test_dense_tdm_codegen(self):
        for dtype in ("fp16", "bf16"):
            for config in (
                (17, 63, 32, 128, 2, False),
                (129, 256, 64, 128, 4, False),
                (129, 257, 128, 128, 4, False),
                (257, 513, 128, 64, 4, False),
                (257, 513, 128, 64, 4, True),
                (1024, 2048, 256, 64, 4, True),
            ):
                with self.subTest(dtype=dtype, config=config):
                    _check_dense_codegen(dtype, *config)


if __name__ == "__main__":
    unittest.main()
