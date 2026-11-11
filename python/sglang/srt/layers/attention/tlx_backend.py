# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in gfx1250 uTLX attention using AITER's cache and metadata lifecycle."""

import inspect
import logging

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

logger = logging.getLogger(__name__)


class TlxAttnBackend(AiterAttnBackend):
    def __init__(self, model_runner, **kwargs):
        if torch.version.hip is None or not torch.cuda.get_device_properties(
            model_runner.device
        ).gcnArchName.startswith("gfx1250"):
            raise ValueError("The tlx attention backend requires AMD gfx1250")
        if model_runner.use_mla_backend:
            raise ValueError("The tlx attention backend supports MHA/GQA, not MLA")
        # The optional plugin is loaded only when this backend is requested.
        from sglang.kernels.ops.attention.tlx.paged import (
            unified_attention,
            unsupported_reason,
        )

        self._tlx_attention = unified_attention
        self._tlx_parameters = frozenset(
            inspect.signature(unified_attention).parameters
        )
        self._tlx_unsupported_reason = unsupported_reason
        self._tlx_fallback_reasons = set()
        self._tlx_dispatched = set()
        super().__init__(model_runner, force_unified_attention=True, **kwargs)

    def _run_unified_attention(self, **kwargs):
        # AITER-only tuning arguments have no uTLX equivalent; keep AITER for them.
        extra = sorted(kwargs.keys() - self._tlx_parameters)
        reason = (
            f"AITER-only arguments {extra}"
            if extra
            else self._tlx_unsupported_reason(**kwargs)
        )
        if reason is None:
            self._log_dispatch(kwargs)
            return self._tlx_attention(**kwargs)
        if reason not in self._tlx_fallback_reasons:
            logger.info("uTLX attention falls back to AITER: %s", reason)
            self._tlx_fallback_reasons.add(reason)
        return super()._run_unified_attention(**kwargs)

    def _log_dispatch(self, kwargs):
        # Host-side metadata only; under graph capture this runs once per captured shape.
        key = (
            kwargs["max_seqlen_q"],
            tuple(kwargs["window_size"]),
            kwargs["q"].dtype,
            kwargs["k"].dtype,
            kwargs["q"].shape[1:],
            kwargs["k"].shape[1:],
        )
        if key not in self._tlx_dispatched:
            self._tlx_dispatched.add(key)
            logger.info(
                "uTLX attention: max_seqlen_q=%d window=%s q=%s%s kv=%s%s",
                key[0],
                key[1],
                key[2],
                tuple(key[4]),
                key[3],
                tuple(key[5]),
            )
