from typing import Callable, Dict, Tuple

import poptorch
import torch
import torch.nn as nn

from flash_attention_ipu import serialised_attention

import pytest


def attention_ref(qkv):
    q, k, v = qkv
    mask = torch.full((q.shape[1], q.shape[1]), -10000)
    mask = torch.triu(mask, 1)
    attn = q @ k.permute(0, 2, 1) + mask
    attn = torch.nn.functional.softmax(attn, dim=-1)
    return attn @ v


def run_forward(
    fn: Callable[..., Dict[str, torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    device: str,
    patterns: Dict[str, bool] = {},
) -> Dict[str, torch.Tensor]:
    class TestModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for k, v in inputs.items():
                self.register_parameter(k, nn.Parameter(v.clone()))

        def forward(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            outputs = fn(**{k: getattr(self, k) for k in inputs})
            return outputs

    module = TestModule()
    if device == "ipu":
        options = poptorch.Options()
        options.useIpuModel(not poptorch.ipuHardwareIsAvailable())
        options._popart.setPatterns(patterns)
        step = poptorch.inferenceModel(module, options)
        output = step()
        step.copyWeightsToHost()
    else:
        output = module()
    with torch.no_grad():
        return dict(**output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_len", [256, 1024, 4096, 16384])
def test_serialised_attention(dtype, seq_len) -> None:
    torch.manual_seed(1123581321)
    N, D = 4, 128
    qkv = torch.randn(3, N, seq_len, D)

    output_ipu = run_forward(
        fn=lambda qkv: dict(out=serialised_attention(qkv, 16, 16)),
        inputs=dict(qkv=qkv),
        device="ipu",
    )

    output_cpu = run_forward(
        fn=lambda qkv: dict(out=attention_ref(qkv)), inputs=dict(qkv=qkv), device="cpu"
    )

    atol = {torch.float32: 1e-4, torch.float16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]

    torch.testing.assert_close(
        output_ipu["out"],
        output_cpu["out"],
        rtol=rtol,
        atol=atol,
    )
