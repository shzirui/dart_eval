#!/usr/bin/env python3
"""Linearly merge two Hugging Face safetensors checkpoints.

Default use in this repository:

    python tools/merge_hf_safetensors.py \
        --model-a models/dart-gui-7b \
        --model-b models/UI-TARS-1.5-7B \
        --output models/dart-gui-uitars-0.5-0.5 \
        --weight-a 0.5 \
        --weight-b 0.5

The merge is done shard by shard, so it does not load the full model into
memory at once. Non-weight files such as tokenizer/config files are copied
from --copy-from, which defaults to model-a.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


WEIGHT_INDEX = "model.safetensors.index.json"
safe_open = None
save_file = None


def require_safetensors() -> None:
    global safe_open, save_file
    if safe_open is not None and save_file is not None:
        return
    try:
        from safetensors import safe_open as imported_safe_open
        from safetensors.torch import save_file as imported_save_file
    except ImportError as exc:  # pragma: no cover - exercised by users' envs
        raise SystemExit(
            "Missing dependency: safetensors. Install the repository requirements "
            "or run `pip install safetensors` in the environment used for merging."
        ) from exc
    safe_open = imported_safe_open
    save_file = imported_save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly merge two sharded Hugging Face safetensors models."
    )
    parser.add_argument("--model-a", type=Path, default=Path("models/dart-gui-7b"))
    parser.add_argument("--model-b", type=Path, default=Path("models/UI-TARS-1.5-7B"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dart-gui-uitars-0.5-0.5"),
        help="Output Hugging Face model directory.",
    )
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument(
        "--copy-from",
        choices=("a", "b"),
        default="a",
        help="Which model directory supplies tokenizer/config/non-weight files.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Temporary dtype used while interpolating floating tensors.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("source", "float32", "bfloat16", "float16"),
        default="source",
        help="Dtype for saved floating tensors. 'source' preserves model-a dtype per tensor.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate compatibility and print the planned merge.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / WEIGHT_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    return read_json(index_path)


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep only structural config fields that must match for weight averaging."""
    keep = [
        "architectures",
        "hidden_size",
        "intermediate_size",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "tie_word_embeddings",
        "vocab_size",
    ]
    out = {k: config.get(k) for k in keep}
    vision = config.get("vision_config") or {}
    out["vision_config"] = {
        k: vision.get(k)
        for k in [
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "out_hidden_size",
            "patch_size",
            "spatial_merge_size",
            "spatial_patch_size",
            "temporal_patch_size",
        ]
    }
    return out


def validate_configs(model_a: Path, model_b: Path) -> None:
    config_a = read_json(model_a / "config.json")
    config_b = read_json(model_b / "config.json")
    if canonical_config(config_a) != canonical_config(config_b):
        raise ValueError(
            "Model configs are not structurally compatible. "
            "Check architectures, hidden sizes, layer counts, vocab size, and vision config."
        )


def validate_indexes(index_a: dict[str, Any], index_b: dict[str, Any]) -> None:
    map_a = index_a.get("weight_map")
    map_b = index_b.get("weight_map")
    if not isinstance(map_a, dict) or not isinstance(map_b, dict):
        raise ValueError("Both index files must contain a weight_map object.")
    keys_a = set(map_a)
    keys_b = set(map_b)
    if keys_a != keys_b:
        missing_in_b = sorted(keys_a - keys_b)[:10]
        missing_in_a = sorted(keys_b - keys_a)[:10]
        raise ValueError(
            "Model weight keys differ. "
            f"Missing in model-b: {missing_in_b}; missing in model-a: {missing_in_a}"
        )
    if map_a != map_b:
        raise ValueError(
            "The two models have the same tensor keys but different shard placement. "
            "Re-shard one model first, or extend this script to remap tensors across shards."
        )


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def copy_non_weight_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name == WEIGHT_INDEX or item.suffix == ".safetensors":
            continue
        target = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def validate_shard_metadata(model_a: Path, model_b: Path, shards: dict[str, list[str]]) -> None:
    """Validate shard contents and shapes without materializing tensor data."""
    for shard_name, keys in shards.items():
        path_a = model_a / shard_name
        path_b = model_b / shard_name
        if not path_a.is_file():
            raise FileNotFoundError(f"Missing shard in model-a: {path_a}")
        if not path_b.is_file():
            raise FileNotFoundError(f"Missing shard in model-b: {path_b}")

        with safe_open(path_a, framework="pt", device="cpu") as fa, safe_open(
            path_b, framework="pt", device="cpu"
        ) as fb:
            keys_a = set(fa.keys())
            keys_b = set(fb.keys())
            expected = set(keys)
            if keys_a != expected:
                raise ValueError(f"Shard keys do not match index for model-a {shard_name}")
            if keys_b != expected:
                raise ValueError(f"Shard keys do not match index for model-b {shard_name}")
            for key in keys:
                shape_a = tuple(fa.get_slice(key).get_shape())
                shape_b = tuple(fb.get_slice(key).get_shape())
                if shape_a != shape_b:
                    raise ValueError(f"Shape mismatch for {key}: {shape_a} vs {shape_b}")


def merge_tensor(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    weight_a: float,
    weight_b: float,
    compute_dtype: torch.dtype,
    output_dtype_name: str,
) -> torch.Tensor:
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Tensor shape mismatch: {tensor_a.shape} vs {tensor_b.shape}")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError(f"Tensor dtype mismatch: {tensor_a.dtype} vs {tensor_b.dtype}")
    if not tensor_a.dtype.is_floating_point:
        raise TypeError(f"Non-floating tensor is not supported for averaging: {tensor_a.dtype}")

    merged = tensor_a.to(compute_dtype).mul_(weight_a).add_(tensor_b.to(compute_dtype), alpha=weight_b)
    if output_dtype_name == "source":
        return merged.to(tensor_a.dtype)
    return merged.to(dtype_from_name(output_dtype_name))


def merge_shard(
    model_a: Path,
    model_b: Path,
    output_dir: Path,
    shard_name: str,
    keys: list[str],
    weight_a: float,
    weight_b: float,
    compute_dtype: torch.dtype,
    output_dtype_name: str,
) -> None:
    merged_tensors: dict[str, torch.Tensor] = {}
    with safe_open(model_a / shard_name, framework="pt", device="cpu") as fa, safe_open(
        model_b / shard_name, framework="pt", device="cpu"
    ) as fb:
        for key in keys:
            merged_tensors[key] = merge_tensor(
                fa.get_tensor(key),
                fb.get_tensor(key),
                weight_a,
                weight_b,
                compute_dtype,
                output_dtype_name,
            )
    save_file(merged_tensors, output_dir / shard_name, metadata={"format": "pt"})


def update_output_config(output_dir: Path, output_dtype_name: str, weight_map: dict[str, str]) -> None:
    config_path = output_dir / "config.json"
    if not config_path.is_file() or output_dtype_name == "source":
        return
    config = read_json(config_path)
    config["torch_dtype"] = output_dtype_name
    vision_config = config.get("vision_config")
    if isinstance(vision_config, dict):
        vision_config["torch_dtype"] = output_dtype_name
    write_json(config_path, config)


def write_merge_info(
    output_dir: Path,
    args: argparse.Namespace,
    index_a: dict[str, Any],
) -> None:
    info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "linear",
        "formula": "output = weight_a * model_a + weight_b * model_b",
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "weight_a": args.weight_a,
        "weight_b": args.weight_b,
        "copy_from": args.copy_from,
        "compute_dtype": args.compute_dtype,
        "output_dtype": args.output_dtype,
        "num_tensors": len(index_a["weight_map"]),
        "shards": sorted(set(index_a["weight_map"].values())),
    }
    write_json(output_dir / "merge_info.json", info)


def main() -> None:
    args = parse_args()
    for model_dir in (args.model_a, args.model_b):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Output directory already exists: {args.output}. Use --force to overwrite.")

    validate_configs(args.model_a, args.model_b)
    index_a = load_index(args.model_a)
    index_b = load_index(args.model_b)
    validate_indexes(index_a, index_b)
    weight_map: dict[str, str] = index_a["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        shards.setdefault(shard_name, []).append(key)
    require_safetensors()
    validate_shard_metadata(args.model_a, args.model_b, shards)

    print(
        f"Validated {len(weight_map)} tensors across {len(shards)} shards. "
        f"Merge weights: {args.weight_a:g} / {args.weight_b:g}."
    )
    if args.dry_run:
        for shard_name in sorted(shards):
            print(f"  {shard_name}: {len(shards[shard_name])} tensors")
        return

    if args.output.exists() and args.force:
        shutil.rmtree(args.output)

    copy_source = args.model_a if args.copy_from == "a" else args.model_b
    copy_non_weight_files(copy_source, args.output)

    compute_dtype = dtype_from_name(args.compute_dtype)
    for shard_name in sorted(shards):
        print(f"Merging {shard_name} ({len(shards[shard_name])} tensors)")
        merge_shard(
            args.model_a,
            args.model_b,
            args.output,
            shard_name,
            sorted(shards[shard_name]),
            args.weight_a,
            args.weight_b,
            compute_dtype,
            args.output_dtype,
        )

    write_json(args.output / WEIGHT_INDEX, index_a)
    update_output_config(args.output, args.output_dtype, weight_map)
    write_merge_info(args.output, args, index_a)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
