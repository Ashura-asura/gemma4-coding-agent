"""Convert the PEFT SFT adapter to a llama.cpp GGUF LoRA.

The official ``convert_lora_to_gguf.py`` needs the ``conversion`` package
from the llama.cpp source tree. This reimplements only what this adapter
uses: linear LoRA pairs mapped onto the exact tensor names present in the
base GGUF. It aborts on any unmapped or missing tensor rather than guessing.

Usage:
    python scripts/make_lora_gguf.py \
        --adapter <peft-dir> --base <base.gguf> --out <lora.gguf>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gguf
import torch
from safetensors.torch import load_file

DEFAULT_ADAPTER = Path(
    os.environ.get(
        "GEMMA_ADAPTER",
        r"C:\Users\bisha\.cache\kagglehub\datasets\ashuraasura\sft-adapter\versions\1\sft",
    )
)
DEFAULT_BASE = Path(
    os.environ.get(
        "GEMMA_GGUF",
        r"C:\Users\bisha\.cache\gemma-local\gguf\gemma-4-E4B-it-Q3_K_M.gguf",
    )
)
DEFAULT_OUT = Path(
    os.environ.get(
        "GEMMA_LORA",
        r"C:\Users\bisha\.cache\gemma-local\lora\sft-lora-f16.gguf",
    )
)

MODULE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    alpha = float(cfg["lora_alpha"])
    rank = int(cfg["r"])
    print(f"adapter r={rank} alpha={alpha} scale={alpha / rank}")

    reader = gguf.GGUFReader(str(args.base))
    base_names = {t.name for t in reader.tensors}
    arch = reader.fields["general.architecture"].contents()
    print(f"base arch={arch} tensors={len(base_names)}")

    state = load_file(str(args.adapter / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        if name.endswith(".lora_A.weight"):
            pairs.setdefault(name[: -len(".lora_A.weight")], {})["A"] = tensor
        elif name.endswith(".lora_B.weight"):
            pairs.setdefault(name[: -len(".lora_B.weight")], {})["B"] = tensor
        else:
            print(f"UNEXPECTED tensor {name} {tuple(tensor.shape)}")
            return 1

    prefix = "model.language_model.layers."
    mapped: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    problems: list[str] = []
    for hf_name, pair in sorted(pairs.items()):
        hf_name = hf_name.removeprefix("base_model.model.")
        if "A" not in pair or "B" not in pair:
            problems.append(f"incomplete pair {hf_name}")
            continue
        if not hf_name.startswith(prefix):
            problems.append(f"unexpected prefix {hf_name}")
            continue
        rest = hf_name[len(prefix):]
        layer, _, module = rest.partition(".")
        short = MODULE_MAP.get(module)
        if short is None:
            problems.append(f"unmapped module {module}")
            continue
        dest = f"blk.{layer}.{short}.weight"
        if dest not in base_names:
            problems.append(f"{dest} missing from base")
            continue
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            problems.append(f"bad shapes {hf_name} A={tuple(a.shape)} B={tuple(b.shape)}")
            continue
        mapped.append((dest, a, b))

    if problems:
        print(f"REFUSING: {len(problems)} problems")
        for problem in problems[:20]:
            print(" ", problem)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch=str(arch), use_temp_file=False)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    for dest, a, b in mapped:
        writer.add_tensor(dest + ".lora_a", a.to(torch.float16).numpy())
        writer.add_tensor(dest + ".lora_b", b.to(torch.float16).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = args.out.stat().st_size
    print(f"WROTE {len(mapped)} pairs -> {args.out} ({size / 2**20:.1f} MiB)")
    print("sample:", mapped[0][0], tuple(mapped[0][1].shape), tuple(mapped[0][2].shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
