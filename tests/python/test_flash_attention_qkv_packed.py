# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable, Dict, Tuple

import poptorch
import torch
import torch.nn as nn

from flash_attention_ipu import flash_attention_qkv_packed

import pytest


def run_forward_and_backward(
    fn: Callable[..., Dict[str, torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    device: str,
    grad_outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    class TestModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for k, v in inputs.items():
                self.register_parameter(k, nn.Parameter(v.clone()))

        def forward(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            outputs = fn(**{k: getattr(self, k) for k in inputs})
            loss = poptorch.identity_loss(
                sum(
                    torch.sum(v * grad_outputs.get(k, torch.ones_like(v)).to(v.device))
                    for k, v in outputs.items()
                ),
                reduction="none",
            )
            return outputs, loss

    module = TestModule()
    optimiser = torch.optim.SGD(module.parameters(), 1.0)
    if device == "ipu":
        options = poptorch.Options()
        options.useIpuModel(not poptorch.ipuHardwareIsAvailable())
        step = poptorch.trainingModel(module, options, optimiser)
        output, _ = step()
        step.copyWeightsToHost()
    else:
        optimiser.zero_grad()
        output, loss = module()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return dict(
            **output, **{f"grad_{k}": inputs[k] - getattr(module, k) for k in inputs}
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_len", [256, 1024, 4096, 16384])
def test_flash_attention_qkv_packed(dtype, seq_len) -> None:
    torch.manual_seed(1123581321)
    N, D = 4, 128
    qkv = torch.randn(3, N, seq_len, D)
    grad = torch.randn(N, seq_len, D)

    output_ipu = run_forward_and_backward(
        fn=lambda qkv: dict(out=flash_attention_qkv_packed(qkv, 16, 16)),
        inputs=dict(qkv=qkv),
        grad_outputs=dict(out=grad),
        device="ipu",
    )

    output_cpu = run_forward_and_backward(
        fn=lambda qkv: dict(out=flash_attention_qkv_packed(qkv, 1, 1)),
        inputs=dict(qkv=qkv),
        grad_outputs=dict(out=grad),
        device="cpu",
    )

    atol = {torch.float32: 1e-3, torch.float16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]

    torch.testing.assert_close(
        output_ipu["out"],
        output_cpu["out"],
        rtol=rtol,
        atol=atol,
    )

    torch.testing.assert_close(
        output_ipu["grad_qkv"],
        output_cpu["grad_qkv"],
        rtol=rtol,
        atol=atol,
    )
