# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend KV-cache dtype handlers registered via vLLM's
``register_kv_cache_dtype`` mechanism.

Importing this module eagerly decorates the handlers, which appends each name
to ``KV_CACHE_DTYPES``, injects the torch dtype into
``STR_DTYPE_TO_TORCH_DTYPE`` in place, and stores the handler for later
``quant_mode()`` / ``is_quantized()`` queries. The in-place dict mutation means
that once this module is imported (which happens in every process that builds a
VllmConfig, via the platform's quant-config registration), the upstream
``kv_cache_dtype_str_to_dtype`` lookup resolves ``"fp8"`` to
``torch.float8_e4m3fn`` rather than the upstream ``torch.uint8``.
"""

import torch

try:
    # vllm >= the version that ships the pluggable kv-cache-dtype mechanism
    # (e.g. the kvquant_27 branch / future main). Importing runs the
    # @register_kv_cache_dtype decorators below, which append the name to
    # KV_CACHE_DTYPES and inject the Ascend torch dtype into the per-process
    # STR_DTYPE_TO_TORCH_DTYPE dict in place.
    from vllm.config.cache import register_kv_cache_dtype
except ImportError:
    # vllm without the pluggable mechanism (e.g. releases/v0.27.1): there is
    # no register_kv_cache_dtype. Registration is skipped and the fp8 dtype
    # flip is performed instead by
    # vllm_ascend/patch/worker/patch_kv_cache_dtype.py (which mutates
    # STR_DTYPE_TO_TORCH_DTYPE["fp8"] -> torch.float8_e4m3fn directly).
    # Fp8AscendHandler / Int8AscendHandler are still defined as plain classes
    # so vllm_ascend/worker/worker.py's `import ... # noqa: F401` keeps
    # resolving on every vllm version.
    register_kv_cache_dtype = None

from vllm.v1.kv_cache_interface import KVQuantMode


def _noop_register(name: str):
    """Identity decorator used when register_kv_cache_dtype is unavailable.

    Keeps the class definitions below unchanged across vllm versions while
    skipping the actual (impossible) registration on old vllm.
    """

    def _decorate(cls):
        return cls

    return _decorate


_register = register_kv_cache_dtype if register_kv_cache_dtype is not None else _noop_register


@_register("fp8")
class Fp8AscendHandler:
    """fp8 KV cache stored as ``torch.float8_e4m3fn`` on Ascend.

    The upstream builtin ``"fp8"`` maps to ``torch.uint8`` (raw fp8 bytes
    consumed by NVIDIA cutlass kernels). Ascend's MLA/DSA kernels operate on
    native ``torch.float8_e4m3fn`` directly, so re-registering ``"fp8"`` here
    flips that mapping to ``float8_e4m3fn``. The decorator logs an
    ``already exists and will be overwritten`` warning; this is expected and
    intentional on the Ascend platform.

    No per-token scale: ``quant_mode`` returns ``KVQuantMode.BACKEND`` so the
    Ascend backend fully self-manages kernel dispatch rather than reusing an
    upstream generic fp8/per-token-head kernel path.
    """

    name = "fp8"

    def torch_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def is_quantized(self) -> bool:
        return True

    def quant_mode(self) -> KVQuantMode:
        return KVQuantMode.BACKEND


@_register("int8")
class Int8AscendHandler:
    """int KV cache stored as ``torch.int8`` on Ascend.
    """

    name = "int8"

    def torch_dtype(self) -> torch.dtype:
        return torch.int8

    def is_quantized(self) -> bool:
        return True

    def quant_mode(self) -> KVQuantMode:
        return KVQuantMode.BACKEND
