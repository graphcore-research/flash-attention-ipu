from typing import Callable, Optional
import torch
import poptorch
from flash_attention_qkv_packed import flash_attention_qkv_packed
from utils import patch_function
from math import log2, floor, ceil


@patch_function(torch.nn.functional.scaled_dot_product_attention, [torch])
def scaled_dot_product_attention_patch(
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
            "flash_attention_ipu does not currently support Grouped- or Multi-query attention"
        )
    if key.shape != value.shape:
        raise NotImplementedError(
            "flash_attention_ipu does not currently support value.shape != key.shape"
        )
    if poptorch.isRunningOnIpu():
        N, H, L, D = query.shape

        num_chunks_q = int(2 ** (ceil(log2(L // D) / 2)))
        num_chunks_kv = int(2 ** (floor(log2(L // D) / 2)))
        query *= D**0.5
        qkv = torch.stack([query, key, value])
        out = flash_attention_qkv_packed(
            qkv.reshape(N * H, L, D), num_chunks_q, num_chunks_kv
        )
        return out.reshape(N, H, L, D)
    else:
        return orig_fn(query, key, value, attn_mask, dropout_p, is_causal)
