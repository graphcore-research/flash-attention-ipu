import poptorch
import torch


def serialised_attention(qkv: torch.Tensor, num_chunks_q: int, num_chunks_kv: int):
    if qkv.ndim != 4:
        raise ValueError("serialed_attention expects qkv input to have 4 dimensions")
    if qkv.shape[0] != 3:
        raise ValueError(
            "serialised_attention expects qkv input to have size 3 at dimension 0"
        )
    if qkv.shape[2] % num_chunks_q != 0:
        raise ValueError(
            "serialised_attention expects qkv size at dimension 2 to be divisible by num_chunks_q"
        )
    if qkv.shape[2] % num_chunks_kv != 0:
        raise ValueError(
            "serialised_attention expects qkv size at dimension 2 to be divisible by num_chunks_kv"
        )

    out: torch.Tensor
    if poptorch.isRunningOnIpu():
        (out,) = poptorch.custom_op(
            name="SerialisedAttention",
            domain_version=1,
            domain="ai.graphcore",
            inputs=[qkv],
            example_outputs=[qkv[0]],
            attributes={"num_chunks_q": num_chunks_q, "num_chunks_kv": num_chunks_kv},
        )
    else:
        q, k, v = qkv
        attn = q @ k.permute(0, 2, 1)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = attn @ v

    return out


__all__ = ["serialised_attention"]
