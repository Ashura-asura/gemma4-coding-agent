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
HF_SLUG = "google/gemma-4-12B-it"
MODEL_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM")


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def resolve_base_model(config: dict[str, Any]) -> str:
    """Config value > Kaggle model download > HF hub (§0.3: 12B-it primary)."""
    if config.get("base_model"):
        return str(config["base_model"])
    if _is_kaggle():
        import kagglehub

        return kagglehub.model_download(KAGGLE_SLUG)
    return HF_SLUG


def _dtype(name: str, torch: Any) -> Any:
    wanted = getattr(torch, name)
    if wanted == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
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

    errors: list[str] = []
    for name in MODEL_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base, **kwargs)
            print(f"[sft] loaded with {name}", flush=True)
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
    base = base_override or resolve_base_model(config)
    print(f"[sft] base model: {base}", flush=True)

    model, processor = load_model_and_processor(base, config)
    dataset = build_dataset(config["data"]["path"], processor, config)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from train.sft.data_collator import TrajectoryCollator

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"])
    )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
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

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_arguments(config, torch),
        train_dataset=dataset,
        data_collator=TrajectoryCollator(processor),
    )
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except torch.cuda.OutOfMemoryError:
        # §0.3 fallback: rerun with base_model_fallback (E4B) — progress under
        # a different base can't resume, so fail loudly instead of corrupting
        trainer.save_state()
        fallback = config.get("base_model_fallback")
        hint = f" --base-model {fallback}" if fallback else ""
        print(
            f"[sft] CUDA OOM — checkpoint state saved. Rerun with the E4B fallback:\n"
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
