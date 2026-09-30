"""resolve_lora_targets — peft injection keys for Gemma-4 (ARCHITECTURE §2.4).

Failure this pins: peft's get_peft_model raised ``ValueError: Target module
Gemma4ClippableLinear(...) is not supported`` — E4B's vision/audio towers
reuse q/k/v/o projection names inside clippable wrappers (768-wide), and a
bare leaf-name target list matches them before/alongside the language model.
The resolver returns exact keys scoped to ``language_model`` (towers never
see gradients in text-only SFT) and unwraps any wrapped projection to the
inner ``linear`` child peft can actually wrap.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.sft.trainer import resolve_lora_targets  # noqa: E402


class Clippable(torch.nn.Module):
    """Stands in for Gemma4ClippableLinear: wrapper whose child is the real Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(8, 8, bias=False)
        self.second = torch.nn.Linear(8, 8, bias=False)


class LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)  # plain text projection
        self.k_proj = Clippable()  # a wrapped text projection (defensive path)
        self.q_norm = torch.nn.LayerNorm(8)  # non-target leaf, untouched


class Tower(torch.nn.Module):
    """vision/audio tower: same leaf names, clippable wrappers, text never uses them."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = Clippable()
        self.o_proj = Clippable()


class FakeMultimodal(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModel()
        self.vision_tower = Tower()
        self.audio_tower = Tower()


def test_targets_scoped_to_language_model_and_unwrapped():
    model = FakeMultimodal()
    keys = resolve_lora_targets(model, ["q_proj", "k_proj", "o_proj"], torch)
    assert keys == ["language_model.k_proj.linear", "language_model.q_proj"]


def test_ambiguous_wrapper_fails_loudly():
    class Bad(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = LanguageModel()
            self.language_model.q_proj = TwoInner()

    with pytest.raises(SystemExit, match="expected exactly 1"):
        resolve_lora_targets(Bad(), ["q_proj"], torch)


def test_no_matching_leaves_fails_loudly():
    with pytest.raises(SystemExit, match="matched no modules"):
        resolve_lora_targets(FakeMultimodal(), ["nope_proj"], torch)


def test_model_without_language_model_scans_whole_tree():
    class Plain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.nested = torch.nn.Linear(8, 8, bias=False)

    keys = resolve_lora_targets(Plain(), ["q_proj", "nested"], torch)
    assert keys == ["nested", "q_proj"]


def test_tiny_gemma4_end_to_end_wraps_and_trains():
    """Real architecture: resolve keys -> get_peft_model -> backward on LoRA."""
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    torch.manual_seed(0)
    text = Gemma4TextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=512,
        hidden_size_per_layer_input=16,
        max_position_embeddings=512,
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text))
    leaves = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    keys = resolve_lora_targets(model, leaves, torch)
    # 2 layers x 7 projections, all plain Linear in the text model
    assert len(keys) == 14
    assert all("language_model" in key for key in keys)

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=keys,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0

    input_ids = torch.randint(3, 512, (1, 16))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = [
        p.grad
        for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n and p.grad is not None
    ]
    assert grads and any(g.abs().sum() > 0 for g in grads)
