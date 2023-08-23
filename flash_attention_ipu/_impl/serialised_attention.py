import poptorch
import torch
import torch.nn.functional as F
import einops


def serialised_attention(
    qkv: torch.Tensor, num_chunks_q: int, num_chunks_kv: int
) -> torch.Tensor:
    """
    Memory-efficient causally masked multi-head attention from a packed qkv tensor

    Computes `nn.softmax(Q@K.T, dim=-1)@V` without materialising the full attention
    matrix using chunking and online softmax for memory optimisation

    qkv -- shape (3, N, H, L, D)
    returns -- shape (N, H, L, D)

    """
    if qkv.ndim != 5:
        raise ValueError("serialed_attention expects qkv input to have 4 dimensions")
    if qkv.shape[0] != 3:
        raise ValueError(
            "serialised_attention expects qkv input to have size 3 at dimension 0"
        )
    if qkv.shape[3] % num_chunks_q != 0:
        raise ValueError(
            "serialised_attention expects qkv size at dimension 2 to be divisible by num_chunks_q"
        )
    if qkv.shape[3] % num_chunks_kv != 0:
        raise ValueError(
            "serialised_attention expects qkv size at dimension 2 to be divisible by num_chunks_kv"
        )

    out: torch.Tensor
    N, H, L, D = qkv.shape[1:]
    if poptorch.isRunningOnIpu():
        qkv[0] *= D ** (-0.5)
        qkv = einops.rearrange(qkv, "three n h l d -> three (n h) l d")
        (out,) = poptorch.custom_op(
            name="SerialisedAttention",
            domain_version=1,
            domain="ai.graphcore",
            inputs=[qkv],
            example_outputs=[qkv[0]],
            attributes={"num_chunks_q": num_chunks_q, "num_chunks_kv": num_chunks_kv},
        )
        out = einops.rearrange(qkv, "(n h) l d -> n h l d", n=N, h=H)
    else:
        q, k, v = qkv
        if torch.cuda.is_available():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.full((q.shape[-2], q.shape[-2]), -10000)
            mask = torch.triu(mask, 1)
            attn = q @ k.permute(0, 2, 1) * D ** (-0.5) + mask
            attn = torch.nn.functional.softmax(attn, dim=-1)
            out = attn @ v
    return out


__all__ = ["serialised_attention"]
