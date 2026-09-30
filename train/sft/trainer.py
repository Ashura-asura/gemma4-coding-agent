"""QLoRA SFT trainer — ARCHITECTURE.md §2.4.

Designed for a Kaggle free-tier T4/P100 session (§5): 4-bit base + LoRA,
completion-only loss, frequent resumable checkpoints. Entry point:
``python -m train.sft.trainer --config configs/sft_config.yaml``.

Gemma-4 specifics (verified against the model cards): the ``-it`` checkpoints
ship ``chat_template.jinja`` (base does not), the chat template lives on
``AutoProcessor`` (not a bare tokenizer), and the unified arch may register
under any of three auto classes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

KAGGLE_SLUG = "google/gemma-4/transformers/gemma-4-12b-it"
KAGGLE_SLUG_E4B = "google/gemma-4/transformers/gemma-4-e4b-it"
HF_SLUG = "google/gemma-4-12B-it"
HF_SLUG_E4B = "google/gemma-4-E4B-it"
MODEL_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM")


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def resolve_base_model(config: dict[str, Any], *, which: str = "primary") -> str:
    """Config value > Kaggle model download > HF hub (§0.3: 12B primary, E4B fallback)."""
    if which == "primary" and config.get("base_model"):
        return str(config["base_model"])
    fallback = which == "fallback"
    if _is_kaggle():
        models = Path("/kaggle/input/models")
        pattern = "gemma-4-e4b-it/*/config.json" if fallback else "gemma-4-12b-it/*/config.json"
        if models.is_dir():
            mounted = sorted(models.glob(f"**/{pattern}"))
            if mounted:
                return str(mounted[0].parent)
        import kagglehub

        return kagglehub.model_download(KAGGLE_SLUG_E4B if fallback else KAGGLE_SLUG)
    return HF_SLUG_E4B if fallback else HF_SLUG


def _dtype(name: str, torch: Any) -> Any:
    wanted = getattr(torch, name)
    if wanted == torch.bfloat16 and torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        # T4/P100 (§5) have no bf16 tensor cores — fp16 keeps the config's intent
        print("[sft] bf16 unsupported on this GPU; using float16", flush=True)
        return torch.float16
    return wanted


def _load_first_model(base: str, config: dict[str, Any], torch: Any) -> Any:
    """Try the auto classes until one accepts Gemma-4's unified arch."""
    from transformers import BitsAndBytesConfig

    bits = int(config.get("quantization", {}).get("bits", 4))
    compute_dtype = _dtype(str(config.get("quantization", {}).get("compute_dtype", "bfloat16")), torch)
    kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if bits == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        kwargs["torch_dtype"] = compute_dtype

    import transformers

    try:
        import bitsandbytes as _bnb

        print(f"[sft] bitsandbytes {_bnb.__version__}", flush=True)
    except Exception as exc:
        print(f"[sft] bitsandbytes unavailable: {exc}", flush=True)

    errors: list[str] = []
    for name in MODEL_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base, **kwargs)
            print(f"[sft] loaded with {name}", flush=True)
            if bits == 4 and not getattr(model, "is_loaded_in_4bit", False):
                raise SystemExit(
                    "[sft] 4-bit quantization did not engage — training would OOM. "
                    "Check the bitsandbytes pin in pyproject.toml."
                )
            placement = sorted({str(p.device) for p in model.parameters()})
            print(f"[sft] param devices: {placement} | quantized={getattr(model, 'is_loaded_in_4bit', False)}", flush=True)
            return model
        except Exception as exc:  # arch registration differs across cards/versions
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise SystemExit("[sft] no auto model class accepted the checkpoint:\n  " + "\n  ".join(errors))


def load_model_and_processor(base: str, config: dict[str, Any]):
    """4-bit base + processor (Gemma-4's chat template lives on the processor)."""
    import torch
    from transformers import AutoProcessor

    model = _load_first_model(base, config, torch)
    processor = AutoProcessor.from_pretrained(base)
    inner = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if not getattr(inner, "chat_template", None) and not getattr(processor, "chat_template", None):
        raise SystemExit(
            f"[sft] {base} has no chat template — only -it checkpoints ship "
            f"chat_template.jinja; set config base_model accordingly"
        )
    if inner.pad_token_id is None and inner.eos_token_id is not None:
        inner.pad_token = inner.eos_token
    return model, processor


def build_dataset(path: str, processor: Any, config: dict[str, Any]):
    """jsonl trajectories -> pre-tokenized dataset with completion-only labels."""
    from datasets import Dataset

    from train.sft.data_collator import encode_trajectory

    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if len(record.get("messages") or []) >= 2]
    if not records:
        raise SystemExit(f"no trajectories in {path} — finish the data build first (§2.1)")
    max_seq_len = int(config["training"]["max_seq_len"])

    def _encode(record: dict[str, Any]) -> dict[str, list[int]]:
        return encode_trajectory(processor, record["messages"], max_seq_len)

    dataset = Dataset.from_list([{"messages": record["messages"]} for record in records])
    return dataset.map(
        _encode,
        remove_columns=["messages"],
        desc="encoding trajectories",
        num_proc=1 if sys.platform == "win32" else 2,
    )


def prepare_for_qlora(model: Any, torch: Any) -> Any:
    """Freeze the base and fp32-cast only 1-D norms/biases.

    peft's ``prepare_model_for_kbit_training`` upcasts *every* non-quantized
    half-precision parameter to fp32; Gemma-4's per-layer embedding
    ([262144, 10752] = 2.8B params, §0.3) would alone need 10.5 GiB and OOMs
    a 16 GiB T4. Frozen tables stay in their load dtype (they get no
    gradients). Gradient checkpointing is left to the Trainer
    (non-reentrant — see ``training_arguments``), which activates it on the
    PEFT-wrapped model at train() start.
    """
    half = (torch.float16, torch.bfloat16)
    upcast = 0
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1 and param.dtype in half:
            param.data = param.data.to(torch.float32)
            upcast += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[sft] prepare_for_qlora: froze base, fp32-cast {upcast} norm/bias tensors", flush=True)
    return model


def resolve_lora_targets(model: Any, leaves: Any, torch: Any) -> list[str]:
    """Exact module keys for LoRA injection over the policy language model.

    Gemma-4's vision/audio towers reuse the q/k/v/o/gate/up/down projection
    names inside ``Gemma4ClippableLinear`` wrappers, which peft refuses to
    wrap (``ValueError: ... is not supported``); text-only SFT never touches
    those towers, so when the model has a ``language_model`` component every
    other subtree is excluded. A projection that *is* wrapped (future text
    configs may clip) is targeted at its inner ``linear`` child — the concrete
    nn.Linear/Linear4bit peft can actually wrap.
    """
    wanted = {str(leaf) for leaf in leaves}
    modules = list(model.named_modules())
    has_lm = any("language_model" in key.split(".") for key, _ in modules)
    keys: list[str] = []
    for key, module in modules:
        parts = key.split(".")
        if has_lm and "language_model" not in parts:
            continue
        if parts[-1] not in wanted:
            continue
        if isinstance(module, torch.nn.Linear):
            keys.append(key)
            continue
        inner = [
            f"{key}.{child_key}"
            for child_key, child in module.named_modules()
            if child_key and isinstance(child, torch.nn.Linear)
        ]
        if len(inner) != 1:
            raise SystemExit(
                f"[sft] cannot LoRA-target {key}: {type(module).__name__} has "
                f"{len(inner)} Linear children (expected exactly 1)"
            )
        keys.append(inner[0])
    if not keys:
        raise SystemExit(f"[sft] target_modules {sorted(wanted)} matched no modules")
    return sorted(set(keys))


def _ce_chunk(logits_chunk: Any, targets: Any) -> Any:
    import torch.nn.functional as F

    return F.cross_entropy(logits_chunk.float(), targets, ignore_index=-100, reduction="sum")


def chunked_sliced_ce(logits: Any, shift_labels: Any, chunk: int = 512) -> Any:
    """Mean CE over the supervised rows, computed in row-chunks under
    non-reentrant checkpointing.

    The model-internal loss materialises the full fp32 [K, vocab] chain
    (cast + log-softmax + grad): at seq_len 2048 with a high-K sample that
    peaked past 4 GiB and OOM'd kernel v16 (alloc 14.22 GiB, next request
    1.58 GiB = fp32 [1620, 262144] exactly). Chunking under checkpoint
    keeps only the fp16 logits plus one chunk's fp32 working set (~1 GiB)
    regardless of K, while staying numerically equal to
    transformers' ForCausalLMLoss (mean over non-ignored targets).
    """
    from torch.utils.checkpoint import checkpoint
    import torch

    flat = logits.reshape(-1, logits.shape[-1])  # [B*K, V] — CE rows, not the batch axis
    targets = shift_labels.reshape(-1)  # [B*K] (or [K] for B=1)
    rows = int(flat.shape[0])
    sums = [checkpoint(_ce_chunk, flat[s : s + chunk], targets[s : s + chunk], use_reentrant=False)
            for s in range(0, rows, chunk)]
    if not sums:
        return logits.sum() * 0.0
    count = int((targets != -100).sum())
    return torch.stack(sums).sum() / max(count, 1)


from transformers import Trainer  # noqa: E402  (mid-file: keeps --dry-run import path light above)


class SlicedLossTrainer(Trainer):
    """Trainer that computes the completion loss outside the model.

    Keeps ``labels``/``shift_labels`` out of the forward pass so Gemma-4's
    internal full-vocab fp32 loss graph never materialises; the collator's
    sliced ``logits_to_keep`` still bounds the lm_head to supervised rows,
    and ``chunked_sliced_ce`` bounds the CE working set per chunk.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        shift = inputs.pop("shift_labels", None)
        inputs.pop("labels", None)
        outputs = model(**inputs)
        if shift is None:
            raise SystemExit("[sft] collator did not emit shift_labels")
        loss = chunked_sliced_ce(outputs.logits, shift)
        return (loss, outputs) if return_outputs else loss


def training_arguments(config: dict[str, Any], torch: Any) -> Any:
    from transformers import TrainingArguments

    train = config["training"]
    cuda = torch.cuda.is_available()
    bf16 = bool(cuda and _dtype(str(config.get("quantization", {}).get("compute_dtype", "bfloat16")), torch) == torch.bfloat16)
    fp16 = bool(cuda and not bf16)
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=float(train["epochs"]),
        per_device_train_batch_size=int(train["per_device_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        learning_rate=float(train["learning_rate"]),
        lr_scheduler_type=str(train["lr_scheduler"]),
        warmup_ratio=float(train["warmup_ratio"]),
        gradient_checkpointing=bool(train["gradient_checkpointing"]),
        # non-reentrant: recomputes without the input-require-grads hook,
        # which a frozen base would otherwise need
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=float(train["max_grad_norm"]),
        seed=int(train["seed"]),
        bf16=bf16,
        fp16=fp16,
        optim="adamw_bnb_8bit" if cuda else "adamw_torch",
        logging_steps=int(config["logging_every_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_every_steps"]),
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


def last_checkpoint(output_dir: str) -> str | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def train(config_path: str, *, resume: str | None, base_override: str | None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[sft] training needs a GPU (ARCHITECTURE §5) — use --dry-run for a CPU check")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if base_override == "fallback":
        base = resolve_base_model(config, which="fallback")
    else:
        base = base_override or resolve_base_model(config)
    print(f"[sft] base model: {base}", flush=True)

    model, processor = load_model_and_processor(base, config)
    dataset = build_dataset(config["data"]["path"], processor, config)

    from peft import LoraConfig, get_peft_model

    from train.sft.data_collator import TrajectoryCollator

    model = prepare_for_qlora(model, torch)
    lora = config["lora"]
    targets = resolve_lora_targets(model, lora["target_modules"], torch)
    print(f"[sft] LoRA targets: {len(targets)} modules", flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    resume_from = resume or config["training"].get("resume_from")
    if resume_from in (None, "", "null"):
        resume_from = None
    elif resume_from == "auto":
        resume_from = last_checkpoint(config["output_dir"])

    trainer = SlicedLossTrainer(
        model=model,
        args=training_arguments(config, torch),
        train_dataset=dataset,
        data_collator=TrajectoryCollator(processor),
    )
    sample = TrajectoryCollator(processor)([dataset[0]])
    keep = sample.get("logits_to_keep")
    print(
        f"[sft] batch keys={sorted(sample)} K={keep.numel() if keep is not None else 'FULL'} | "
        f"gradient_checkpointing={trainer.model.is_gradient_checkpointing} | "
        f"cuda alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB",
        flush=True,
    )
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except torch.cuda.OutOfMemoryError as exc:
        # §0.3 fallback: rerun with base_model_fallback (E4B) — progress under
        # a different base can't resume, so fail loudly instead of corrupting
        trainer.save_state()
        print(
            f"[sft] OOM detail: {exc}\n"
            f"[sft] cuda at OOM: alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB",
            file=sys.stderr,
            flush=True,
        )
        fallback = config.get("base_model_fallback")
        hint = " --base-model fallback" if fallback else ""
        print(
            f"[sft] CUDA OOM — checkpoint state saved. Next levers: lower "
            f"training.max_seq_len (now {config['training']['max_seq_len']}), or rerun with "
            f"the E4B fallback:\n"
            f"      python -m train.sft.trainer --config {config_path}{hint}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)

    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])
    print(f"[sft] adapter saved to {config['output_dir']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT (ARCHITECTURE §2.4)")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="auto = latest checkpoint-* in output_dir, or a checkpoint path (§5)",
    )
    parser.add_argument("--base-model", default=None, help="override config base model")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="encode trajectories and report mask stats without loading a model",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.base_model == "fallback":
        base = resolve_base_model(config, which="fallback")
    else:
        base = args.base_model or resolve_base_model(config)

    if args.dry_run:
        from transformers import AutoProcessor

        print(f"[sft] dry-run base: {base}", flush=True)
        processor = AutoProcessor.from_pretrained(base)
        inner = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if inner.pad_token_id is None and inner.eos_token_id is not None:
            inner.pad_token = inner.eos_token
        dataset = build_dataset(config["data"]["path"], processor, config)
        sample = dataset[0]
        trained = sum(1 for label in sample["labels"] if label != -100)
        print(
            f"[sft] dry-run OK: {len(dataset)} trajectories, "
            f"sample tokens={len(sample['input_ids'])}, trained={trained} "
            f"({trained / max(1, len(sample['labels'])):.0%} unmasked)",
            flush=True,
        )
        return

    train(args.config, resume=args.resume, base_override=args.base_model)


if __name__ == "__main__":
    main()
