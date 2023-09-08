from typing import Callable, Dict, Tuple

import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F

orig_fn = F.scaled_dot_product_attention

import flash_attention_ipu.auto

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


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3), (2, 3, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_len", [256, 1024])
def test_scaled_dot_product_attention_vs_cpu(batch_shape, dtype, seq_len) -> None:
    torch.manual_seed(1123581321)
    shape = (*batch_shape, seq_len, 64)
    q, k, v = torch.randn(3, *shape).to(dtype)
    grad = torch.randn(*shape).to(dtype)

    output_ipu = run_forward_and_backward(
        fn=lambda q, k, v: dict(
            out=F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ),
        inputs=dict(q=q, k=k, v=v),
        grad_outputs=dict(out=grad),
        device="ipu",
    )

    output_cpu = run_forward_and_backward(
        fn=lambda q, k, v: dict(
            out=F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ),
        inputs=dict(q=q, k=k, v=v),
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
        output_ipu["grad_q"],
        output_cpu["grad_q"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        output_ipu["grad_k"],
        output_cpu["grad_k"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        output_ipu["grad_v"],
        output_cpu["grad_v"],
        rtol=rtol,
        atol=atol,
    )


def test_out_of_memory_error_is_fixed() -> None:
    torch.manual_seed(1123581321)
    shape = (16, 2048, 64)
    dtype = torch.float32
    q, k, v = torch.randn(3, *shape).to(dtype)
    grad = torch.randn(*shape).to(dtype)

    # Temporarily undo patch
    patch_fn = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = F.scaled_dot_product_attention.__wrapped__
    try:
        run_forward_and_backward(
            fn=lambda q, k, v: dict(
                out=F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ),
            inputs=dict(q=q, k=k, v=v),
            grad_outputs=dict(out=grad),
            device="ipu",
        )
        assert False  # This should go out of memory
    except poptorch.poptorch_core.Error:
        assert True

    # Reapply patch
    F.scaled_dot_product_attention = patch_fn

    try:
        run_forward_and_backward(
            fn=lambda q, k, v: dict(
                out=F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ),
            inputs=dict(q=q, k=k, v=v),
            grad_outputs=dict(out=grad),
            device="ipu",
        )
        assert True  # This should not go out of memory
    except poptorch.poptorch_core.Error:
        assert False


if __name__ == "__main__":
    test_out_of_memory_error_is_fixed()
