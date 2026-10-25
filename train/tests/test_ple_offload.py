"""offload_per_layer_embedding: same outputs as the stock scaled embedding,
table no longer a registered tensor (so .to() can't resurrect it on the GPU)."""
import pytest

torch = pytest.importorskip("torch")

from train.sft.trainer import offload_per_layer_embedding  # noqa: E402  (after importorskip)


class _ScaledEmb(torch.nn.Embedding):
    def __init__(self, n, d, pad, scale):
        super().__init__(n, d, pad)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, ids):
        return super().forward(ids) * self.embed_scale.to(self.weight.dtype)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens_per_layer = _ScaledEmb(64, 24, 0, 4.0)
        self.other = torch.nn.Linear(4, 4)


def test_offload_matches_stock_and_survives_to():
    torch.manual_seed(0)
    m = _Tiny()
    ids = torch.randint(0, 64, (2, 7))
    ref = m.embed_tokens_per_layer(ids).clone()
    nbytes = offload_per_layer_embedding(m, torch, force=True)
    assert nbytes == 64 * 24 * 4
    assert m.embed_tokens_per_layer.weight.numel() == 0  # table is no longer a parameter
    out = m.embed_tokens_per_layer(ids)
    assert torch.equal(out, ref)
    m.to("cpu")  # a blanket .to() must not change anything
    assert torch.equal(m.embed_tokens_per_layer(ids), ref)


def test_missing_ple_is_a_noop():
    assert offload_per_layer_embedding(torch.nn.Linear(2, 2), torch, force=True) == 0
