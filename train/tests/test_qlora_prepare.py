"""prepare_for_qlora — the fp32-upcast guard that keeps E4B trainable on a T4.

peft's prepare_model_for_kbit_training upcasts every non-quantized
half-precision parameter to fp32; Gemma-4's per-layer embedding
([262144, 10752] = 2.8B params) alone would demand 10.5 GiB and OOM a
16 GiB T4 mid-run (ARCHITECTURE §0.3). This pins our replacement's
contract: freeze everything, fp32-cast only 1-D norm/bias tensors.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.sft.trainer import prepare_for_qlora  # noqa: E402


class TinyGemmaLike(torch.nn.Module):
    """Mirrors the shapes that matter: giant 2-D embedding, 2-D linear, 1-D norm."""

    def __init__(self) -> None:
        super().__init__()
        self.per_layer_embedding = torch.nn.Embedding(4096, 512).to(torch.bfloat16)
        self.linear = torch.nn.Linear(64, 64).to(torch.bfloat16)
        self.norm = torch.nn.LayerNorm(64).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


def test_prepare_freezes_all_params():
    model = prepare_for_qlora(TinyGemmaLike(), torch)
    assert all(not p.requires_grad for p in model.parameters())


def test_prepare_upcasts_only_1d_half_tensors():
    model = prepare_for_qlora(TinyGemmaLike(), torch)
    # the tensors peft would upcast to fp32 and OOM on stay in load dtype
    assert model.per_layer_embedding.weight.dtype == torch.bfloat16
    assert model.linear.weight.dtype == torch.bfloat16
    # 1-D norms/biases keep the fp32 upcast for numerically sensitive
    # reductions (bias too — same rule; training runs under autocast, where
    # mixed fp32-bias/bf16-operands linear is handled)
    assert model.linear.bias.dtype == torch.float32
    assert model.norm.weight.dtype == torch.float32
    assert model.norm.bias.dtype == torch.float32


def test_prepare_is_idempotent():
    model = prepare_for_qlora(TinyGemmaLike(), torch)
    again = prepare_for_qlora(model, torch)
    assert again is model
    assert again.norm.weight.dtype == torch.float32
    assert all(not p.requires_grad for p in again.parameters())


def test_prepare_leaves_forward_working():
    # mirror training: Trainer wraps forward in fp16/bf16 autocast, which
    # handles fp32 norms/biases against bf16 operands
    model = prepare_for_qlora(TinyGemmaLike(), torch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(2, 64))
    assert out.shape == (2, 64)
    assert torch.isfinite(out.float()).all()
