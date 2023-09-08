from typing import Callable, Optional
import torch
import poptorch
from .flash_attention_qkv_packed import flash_attention_qkv_packed
from .utils import patch_function
from math import log2, floor, ceil, prod


@patch_function(torch.nn.functional.scaled_dot_product_attention, [torch.nn.functional])
def _scaled_dot_product_attention_ipu(
    orig_fn: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    if attn_mask:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support passing attn_mask"
        )
    if dropout_p > 0.0:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support non-zero dropout_p"
        )
    if not is_causal:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support is_causal=False"
        )
    if query.shape != key.shape:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support Grouped- or Multi-query \
             attention (query.shape != key.shape)"
        )
    if key.shape != value.shape:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support value.shape != key.shape"
        )
    if poptorch.isRunningOnIpu():
        L, D = query.shape[-2:]
        batch_shape = query.shape[:-2]

        # Simple heuristic to choose num_chunks_q and num_chunks_kv
        # General idea: keep attention blocks no larger than queries
        # Choose num_chunks_q and num_chunks_kv such that nelms(q_chunk @ k_chunk.T) <= nelms(q)
        # Use ratio of seq_len and head_dim to decide
        # TODO: make rule more general for non-powers of 2.

        num_chunks_q = int(2 ** (ceil(log2(max(L, D) // D) / 2)))
        num_chunks_kv = int(2 ** (floor(log2(max(L, D) // D) / 2)))

        qkv = torch.stack([query * D**-0.5, key, value])
        out = flash_attention_qkv_packed(
            qkv.reshape(3, prod(batch_shape), L, D), num_chunks_q, num_chunks_kv
        )
        return out.reshape(*batch_shape, L, D)
    else:
        return orig_fn(query, key, value, attn_mask, dropout_p, is_causal)
