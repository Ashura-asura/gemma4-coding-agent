"""GRPO trainer — ARCHITECTURE.md §2.5, §3.3.

    python -m train.rl.grpo --config configs/rl_config.yaml [--resume auto]

Online loop per update: sample ``batch_tasks`` tasks from the RL pool →
roll out *k* trajectories each through the same agent loop/tool interface
as eval → score with the §2.5 reward → group-normalize within the task →
policy-gradient step with k3-KL to the frozen SFT reference (§8.3 guard).

Resumable (§5): adapter + optimizer + update counter saved every
``save_every_updates`` under ``checkpointing.output_dir``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml

from train.sft.data_collator import encode_trajectory

RL_MAX_SEQ_LEN = 4096  # logits for seq×256k vocab must fit a T4 alongside 4bit weights


# ------------------------------------------------------------------ pure math
def token_logprobs(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Per-position log-prob of the label token. Returns (logp, mask) tensors.

    Shifted: logits[t] predicts labels[t+1]; mask excludes -100 and the last
    position (nothing after it).
    """
    import torch

    # bf16 logits stay bf16 until the gather (materializing seq×256k in fp32
    # would add ~4GB next to the 4bit weights on a T4)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:]
    mask = target.ne(-100)
    gathered = logp.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1).float()
    return gathered * mask, mask


def k3_kl(logp: Any, ref_logp: Any, mask: Any) -> Any:
    """Schulman k3 KL estimator token terms: exp(r-p) - (r-p) - 1 >= 0."""
    import torch

    delta = ref_logp - logp
    return (torch.exp(delta) - delta - 1.0) * mask


def grpo_loss(policy_logp: Any, ref_logp: Any, mask: Any, advantage: float, kl_coeff: float) -> Any:
    """Advantage-weighted policy gradient + k3-KL, token-mean normalized (§3.3)."""
    import torch

    tokens = mask.sum().clamp_min(1)
    pg = -(advantage * policy_logp).sum() / tokens
    kl = k3_kl(policy_logp, ref_logp, mask).sum() / tokens
    return pg + kl_coeff * kl, {"pg": float(pg.detach()), "kl": float(kl.detach())}


# ------------------------------------------------------------------- loading
def load_policy(config: dict[str, Any], *, quantize: bool = True) -> tuple[Any, Any]:
    """Trainable policy adapter + frozen SFT reference sharing one 4-bit base."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    policy_path = str(config["policy_checkpoint"])
    if not Path(policy_path).exists():
        raise SystemExit(f"policy checkpoint not found: {policy_path} — run SFT first")
    base_id = PeftConfig.from_pretrained(policy_path).base_model_name_or_path

    kwargs: dict[str, Any] = {}
    if quantize and torch.cuda.is_available():
        dtype = torch.bfloat16
        if str(config.get("training", {}).get("precision", "bfloat16")) == "float16" or (
            torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        ):
            dtype = torch.float16  # T4/P100 have no bf16 tensor cores (§5)
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
            "device_map": "auto",
        }
    else:
        kwargs = {"torch_dtype": "auto"}

    base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    for parameter in base.parameters():
        parameter.requires_grad = False
    policy = PeftModel.from_pretrained(base, policy_path, is_trainable=True)
    ref_path = str(config.get("frozen_sft_baseline") or policy_path)
    if Path(ref_path).exists() and Path(ref_path).resolve() != Path(policy_path).resolve():
        policy.load_adapter(ref_path, adapter_name="ref")
    else:
        # same adapter: snapshot it as the frozen reference before training
        policy.load_adapter(policy_path, adapter_name="ref")
    for name, parameter in policy.named_parameters():
        if ".ref." in name or "reference" in name:
            parameter.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.train()
    policy.set_adapter("policy")
    return policy, tokenizer


# ------------------------------------------------------------------ one step
def rollout_batch(
    tasks: list[dict[str, Any]],
    factory: Any,
    *,
    k: int,
    max_steps: int,
    limits: Any,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    from train.rl.rollout_worker import rollout_task as _rollout_task

    groups = []
    for task in tasks:
        groups.append(
            _rollout_task(task, factory, k=k, max_steps=max_steps, limits=limits, weights=weights)
        )
    return groups


def update_once(
    policy: Any,
    tokenizer: Any,
    groups: list[dict[str, Any]],
    *,
    optimizer: Any,
    kl_coeff: float,
    max_grad_norm: float,
    max_seq_len: int = RL_MAX_SEQ_LEN,
) -> dict[str, float]:
    """One GRPO step over all rollouts of the sampled task batch (§3.3)."""
    import torch

    encoded: list[tuple[list[int], list[int], float]] = []
    for group in groups:
        for rollout, advantage in zip(group["rollouts"], group["normalized"]):
            if not rollout.get("messages"):
                continue
            sample = encode_trajectory(tokenizer, rollout["messages"], max_seq_len)
            encoded.append((sample["input_ids"], sample["labels"], float(advantage)))

    if not encoded:
        return {"loss": 0.0, "pg": 0.0, "kl": 0.0, "sequences": 0}

    total_tokens = sum(sum(1 for label in labels if label != -100) for _, labels, _ in encoded)
    device = next(policy.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    # frozen base + long trajectories: recompute activations instead of storing
    # them (§5: 16GB / 4bit). Off again for rollouts (generate needs the cache).
    policy.gradient_checkpointing_enable()
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    stats = {"pg": 0.0, "kl": 0.0, "sequences": float(len(encoded)), "loss": 0.0}
    try:
        for input_ids, labels, advantage in encoded:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            lab = torch.tensor([labels], dtype=torch.long, device=device)

            policy.set_adapter("ref")
            with torch.no_grad():
                ref_logits = policy(input_ids=ids, use_cache=False).logits
            ref_logp, mask = token_logprobs(ref_logits, lab)
            del ref_logits

            policy.set_adapter("policy")
            logits = policy(input_ids=ids, use_cache=False).logits
            pol_logp, _ = token_logprobs(logits, lab)
            del logits

            # scale each sequence by the batch token count for a stable token-mean
            scale = float(mask.sum()) / max(1, total_tokens)
            loss, loss_stats = _scaled_loss(pol_logp, ref_logp, mask, advantage, kl_coeff, scale)
            loss.backward()
            for key in ("pg", "kl"):
                stats[key] += loss_stats[key] * scale
            stats["loss"] += float(loss.detach())
    finally:
        policy.gradient_checkpointing_disable()

    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_grad_norm
    )
    optimizer.step()
    return stats


def _scaled_loss(
    policy_logp: Any, ref_logp: Any, mask: Any, advantage: float, kl_coeff: float, scale: float
) -> tuple[Any, dict[str, float]]:
    tokens = mask.sum().clamp_min(1)
    pg = -(advantage * policy_logp).sum() / tokens
    kl = k3_kl(policy_logp, ref_logp, mask).sum() / tokens
    loss = (pg + kl_coeff * kl) * scale
    return loss, {"pg": float(pg.detach()), "kl": float(kl.detach())}


# ---------------------------------------------------------------------- cli
def _load_tasks(pool_path: Path) -> list[dict[str, Any]]:
    if not pool_path.exists():
        raise SystemExit(f"missing task pool: {pool_path}")
    return [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _last_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return candidates[-1] if candidates else None


def _dev_tasks(dev_slice: Any, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mid-RL dev gate set (§8.2): a jsonl path or explicit task ids — never the holdout."""
    if not dev_slice:
        return []
    if isinstance(dev_slice, str):
        path = Path(dev_slice)
        if path.exists():
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return []
    wanted = {str(tid) for tid in dev_slice}
    return [task for task in tasks if str(task.get("task_id")) in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRPO policy post-training (ARCHITECTURE §2.5)")
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--resume", default=None, help="auto = latest checkpoint-* in output_dir")
    parser.add_argument("--dry-run", action="store_true", help="pool/config/encode check, no model")
    parser.add_argument("--limit-updates", type=int, help="stop after N updates (smoke runs)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = config["training"]
    rollout_cfg = config.get("rollout") or {}
    ckpt_cfg = config.get("checkpointing") or {}
    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)

    from eval.run_eval import HFPolicy, Limits
    from train.rl.reward import reward_weights
    from train.rl.rollout_worker import rollout_task

    pool_path = Path(rollout_cfg.get("task_pool") or "data/rl_task_pool.jsonl")
    tasks = _load_tasks(pool_path)
    weights = reward_weights(config)
    k = int(rollout_cfg.get("k", 8))
    max_steps = int(rollout_cfg.get("max_steps", 30))
    limits = Limits(wall_seconds=int(rollout_cfg.get("wall_seconds", 600)))

    output_dir = Path(ckpt_cfg.get("output_dir") or "checkpoints/rl")
    resume = args.resume or ckpt_cfg.get("resume_from")
    start_update = 0

    if args.dry_run:
        probe = _tokenizer_probe(config)
        sample = encode_trajectory(probe, _probe_messages(), RL_MAX_SEQ_LEN)
        print(
            f"[grpo] dry-run OK: {len(tasks)} pool tasks, batch_tasks={train_cfg['batch_tasks']}, "
            f"k={k}, probe tokens={len(sample['input_ids'])}, "
            f"trained={sum(1 for x in sample['labels'] if x != -100)}",
            flush=True,
        )
        return 0

    import torch

    policy, tokenizer = load_policy(config)

    if resume in ("auto", "latest"):
        latest = _last_checkpoint(output_dir)
        if latest and (latest / "adapter_config.json").exists():
            from peft import set_peft_model_state_dict
            from peft.utils import load_peft_weights

            state = load_peft_weights(str(latest))
            set_peft_model_state_dict(policy, state, adapter_name="policy")
            state_json = latest / "grpo_state.json"
            if state_json.exists():
                start_update = int(json.loads(state_json.read_text(encoding="utf-8")).get("update", 0))
            print(f"[grpo] resumed from {latest} (update {start_update})", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
    )

    def factory(_task: dict[str, Any]) -> Any:
        # rollouts and logprobs share the one resident model (§5)
        live = HFPolicy(
            max_new_tokens=int(rollout_cfg.get("max_tokens_per_step", 2048)),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            preloaded=(policy, tokenizer),
        )
        return live

    batch_tasks = int(train_cfg["batch_tasks"])
    num_updates = int(train_cfg["num_updates"])
    kl_coeff = float(train_cfg["kl_coeff"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    save_every = int(ckpt_cfg.get("save_every_updates") or 25)
    eval_every = int(ckpt_cfg.get("eval_dev_every_updates") or 0)
    dev_slice_tasks = _dev_tasks(rollout_cfg.get("dev_slice"), tasks)

    print(
        f"[grpo] {len(tasks)} pool tasks; updates {start_update}..{num_updates} "
        f"(batch={batch_tasks}×k={k})",
        flush=True,
    )
    for update in range(start_update, num_updates):
        if args.limit_updates is not None and update >= start_update + args.limit_updates:
            print("[grpo] --limit-updates reached", flush=True)
            break
        batch = random.sample(tasks, min(batch_tasks, len(tasks)))
        started = time.monotonic()
        policy.eval()  # generation must not apply LoRA dropout
        groups = rollout_batch(
            batch, factory, k=k, max_steps=max_steps, limits=limits, weights=weights
        )
        if eval_every and (update + 1) % eval_every == 0 and dev_slice_tasks:
            dev_rewards: list[float] = []
            for dev_task in dev_slice_tasks[:4]:
                dev_group = rollout_task(
                    dev_task, factory, k=1, max_steps=max_steps, limits=limits, weights=weights
                )
                dev_rewards.extend(dev_group["rewards"])
            print(
                f"[grpo] dev@{update + 1}: mean_reward={sum(dev_rewards) / len(dev_rewards):.3f} "
                f"over {len(dev_rewards)} episode(s) (§8.2 mid-RL gate)",
                flush=True,
            )
        policy.train()
        stats = update_once(
            policy,
            tokenizer,
            groups,
            optimizer=optimizer,
            kl_coeff=kl_coeff,
            max_grad_norm=max_grad_norm,
        )
        rewards = [r for g in groups for r in g["rewards"]]
        print(
            f"[grpo] update {update}: loss={stats['loss']:.4f} pg={stats['pg']:.4f} "
            f"kl={stats['kl']:.4f} mean_reward={sum(rewards) / max(1, len(rewards)):.3f} "
            f"best={max(rewards):.3f} ({time.monotonic() - started:.0f}s)",
            flush=True,
        )

        if save_every and (update + 1) % save_every == 0:
            target = output_dir / f"checkpoint-{update + 1}"
            target.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(target)
            (target / "grpo_state.json").write_text(
                json.dumps({"update": update + 1}), encoding="utf-8"
            )
            print(f"[grpo] saved {target}", flush=True)

    policy.save_pretrained(output_dir)
    print(f"[grpo] final adapter -> {output_dir}", flush=True)
    return 0


def _tokenizer_probe(config: dict[str, Any]) -> Any:
    """Dry-run tokenizer: config checkpoint if present, else the -it default."""
    from transformers import AutoTokenizer

    path = str(config.get("policy_checkpoint") or "")
    # a real checkpoint has tokenizer files; the repo placeholder is .gitkeep-only
    if path and any(
        (Path(path) / marker).exists()
        for marker in ("tokenizer_config.json", "tokenizer.json", "spiece.model")
    ):
        return AutoTokenizer.from_pretrained(path)
    from train.sft.trainer import resolve_base_model

    return AutoTokenizer.from_pretrained(resolve_base_model(config))


def _probe_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "submit", "arguments": "{}"}}
        ]},
    ]


if __name__ == "__main__":
    raise SystemExit(main())
