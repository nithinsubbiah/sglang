"""Dense uTLX numerical and alignment coverage; requires gfx1250 hardware."""

import unittest

import torch

from sglang.kernels.ops.attention.tlx.gfx1250 import flash_attn
from sglang.test.test_utils import CustomTestCase


def _reference(q, k, v, scale):
    scores = q.double() @ k.double().transpose(-1, -2) * scale
    return torch.softmax(scores, dim=-1) @ v.double()


class TestTlxDenseAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.version.hip is None
            or not torch.cuda.is_available()
            or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
        ):
            raise unittest.SkipTest("requires gfx1250 and the uTLX TDM extension")

    def assert_accuracy(self, output, reference):
        self.assertTrue(torch.isfinite(output).all().item())
        rms_relative = (
            (output.double() - reference).square().mean() / reference.square().mean()
        ).sqrt()
        self.assertLess(
            rms_relative.item(), 0.02 if output.dtype == torch.bfloat16 else 0.004
        )

    def test_dense_matches_fp64(self):
        for dtype in (torch.float16, torch.bfloat16):
            for sq, sk in (
                (1, 1),
                (17, 63),
                (129, 256),
                (129, 257),
                (129, 513),
                (513, 1025),
            ):
                for fixed in (False, True):
                    with self.subTest(dtype=dtype, sq=sq, sk=sk, fixed=fixed):
                        torch.manual_seed(42)
                        q = torch.randn((1, 2, sq, 128), device="cuda", dtype=dtype)
                        k = torch.randn((1, 2, sk, 128), device="cuda", dtype=dtype)
                        v = torch.randn_like(k)
                        output = flash_attn(
                            q, k, v, sm_scale=0.19, use_fixed_reference=fixed
                        )
                        self.assertEqual(output.shape, q.shape)
                        self.assertEqual(output.dtype, q.dtype)
                        self.assert_accuracy(output, _reference(q, k, v, 0.19))

    def test_fixed_reference_recovers_from_overflow(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = torch.ones((1, 1, 17, 128), device="cuda", dtype=dtype)
                k = torch.ones((1, 1, 513, 128), device="cuda", dtype=dtype)
                k[:, :, :128] = -16
                v = torch.randn_like(k)
                output = flash_attn(q, k, v, use_fixed_reference=True)
                self.assert_accuracy(output, _reference(q, k, v, 128**-0.5))

    def test_aligned_offset_repeated_and_graph_replay(self):
        for dtype in (torch.float16, torch.bfloat16):
            for sk in (127, 513):
                with self.subTest(dtype=dtype, sk=sk):
                    torch.manual_seed(71)

                    def allocate(length):
                        base = torch.randn(
                            8 + 2 * length * 128, device="cuda", dtype=dtype
                        )
                        return base[8:].view(1, 2, length, 128)

                    q, k, v = allocate(129), allocate(sk), allocate(sk)
                    reference = _reference(q, k, v, 128**-0.5)
                    for _ in range(50):
                        self.assert_accuracy(flash_attn(q, k, v), reference)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = flash_attn(q, k, v)
                    for _ in range(50):
                        graph.replay()
                        self.assert_accuracy(output, reference)

    def test_rejects_misaligned_storage_offset(self):
        base = torch.zeros(1 + 129 * 128, device="cuda", dtype=torch.float16)
        q = base[1:].view(1, 1, 129, 128)
        with self.assertRaisesRegex(ValueError, "16-byte aligned"):
            flash_attn(q, q, q)


class TestTlxDenseContract(CustomTestCase):
    def test_rejects_unsupported_inputs(self):
        for causal, dim, scale, match in (
            (True, 128, 0.1, "noncausal"),
            (False, 64, 0.1, "D128"),
            (False, 128, 0.0, "positive finite"),
            (False, 128, float("nan"), "positive finite"),
            (False, 128, 0.1, "one GPU"),
        ):
            with self.subTest(causal=causal, dim=dim, scale=scale):
                q = torch.empty((1, 2, 16, dim), dtype=torch.bfloat16)
                with self.assertRaisesRegex(ValueError, match):
                    flash_attn(q, q, q, causal=causal, sm_scale=scale)


if __name__ == "__main__":
    unittest.main()
