"""
Checkpoint evaluation for Drifting Models.

Computes FID on CIFAR-10 by generating samples and evaluating with clean-fid.
Writes one CSV row per checkpoint and a baseline curve (FID vs step).
"""

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from model import DriftDiT_models
from utils import save_image_grid, set_seed


CSV_COLUMNS = [
    "timestamp_utc",
    "checkpoint",
    "checkpoint_name",
    "dataset",
    "epoch",
    "step",
    "fid",
    "num_samples",
    "batch_size",
    "alpha",
    "seed",
    "loss",
    "drift_norm",
    "grad_norm",
    "feature_dist_mean",
    "feature_dist_var",
    "avg_loss",
    "avg_drift_norm",
    "eval_seconds",
    "artifact_dir",
]


def _dataset_key(name: str) -> str:
    key = name.lower()
    if key == "cifar":
        key = "cifar10"
    return key


def _step_from_name(path: Path) -> int:
    """Best-effort sort key from checkpoint filename."""
    m = re.search(r"step(\d+)", path.stem)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch(\d+)", path.stem)
    if m:
        return int(m.group(1))
    if path.stem.endswith("final"):
        return 10**12
    return -1


def _read_existing_checkpoints(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("checkpoint", "") for row in reader}


def _get_nested_metrics(checkpoint: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}, {}
    step_metrics = metrics.get("step", {})
    epoch_metrics = metrics.get("epoch", {})
    if not isinstance(step_metrics, dict):
        step_metrics = {}
    if not isinstance(epoch_metrics, dict):
        epoch_metrics = {}
    return step_metrics, epoch_metrics


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric(step_metrics: Dict[str, Any], epoch_metrics: Dict[str, Any], name: str, epoch_fallback: Optional[str] = None) -> float:
    if name in step_metrics:
        return _float_or_nan(step_metrics.get(name))
    if epoch_fallback and epoch_fallback in epoch_metrics:
        return _float_or_nan(epoch_metrics.get(epoch_fallback))
    return float("nan")


def discover_checkpoints(checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> List[Path]:
    if checkpoint:
        paths = [Path(checkpoint)]
    elif checkpoint_dir:
        base = Path(checkpoint_dir)
        if not base.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {base}")
        paths = sorted(base.glob("checkpoint*.pt"), key=_step_from_name)
    else:
        raise ValueError("Provide --checkpoint or --checkpoint_dir")

    resolved = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        resolved.append(p.resolve())
    return resolved


def build_model(
    checkpoint: Dict[str, Any],
    dataset: str,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    dataset_key = _dataset_key(dataset)
    if dataset_key == "mnist":
        default_model = "DriftDiT-Tiny"
        default_in_channels = 1
    else:
        default_model = "DriftDiT-Small"
        default_in_channels = 3

    model_name = config.get("model", default_model)
    img_size = int(config.get("img_size", 32))
    in_channels = int(config.get("in_channels", default_in_channels))
    num_classes = int(config.get("num_classes", 10))
    label_dropout = float(config.get("label_dropout", 0.0))

    model_fn = DriftDiT_models[model_name]
    model = model_fn(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        label_dropout=label_dropout,
    ).to(device)

    if use_ema and "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()

    model_info = {
        "model_name": model_name,
        "img_size": img_size,
        "in_channels": in_channels,
        "num_classes": num_classes,
    }
    return model, model_info


@torch.no_grad()
def generate_cifar10_images(
    model: torch.nn.Module,
    out_dir: Path,
    num_samples: int,
    batch_size: int,
    in_channels: int,
    img_size: int,
    num_classes: int,
    alpha: float,
    device: torch.device,
    save_preview: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_batch = None

    index = 0
    while index < num_samples:
        current_batch = min(batch_size, num_samples - index)
        noise = torch.randn(current_batch, in_channels, img_size, img_size, device=device)
        # Class-balanced labels for reproducible evaluation batches.
        labels = (torch.arange(current_batch, device=device) + index) % num_classes
        samples = model.forward_with_cfg(noise, labels, alpha=alpha).clamp(-1, 1)

        if preview_batch is None:
            preview_batch = samples[: min(100, current_batch)].detach().cpu()

        images = ((samples + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        if in_channels == 1:
            images = images.repeat(1, 3, 1, 1)

        for i in range(current_batch):
            image = images[i].permute(1, 2, 0).numpy()
            Image.fromarray(image).save(out_dir / f"{index + i:06d}.png")

        index += current_batch

    if save_preview and preview_batch is not None:
        save_image_grid(preview_batch, str(out_dir.parent / "preview.png"), nrow=10)


def compute_cifar10_fid(generated_dir: Path, split: str = "train", mode: str = "clean") -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError(
            "cleanfid is required for FID evaluation. Install `clean-fid`."
        ) from exc

    return float(
        fid.compute_fid(
            str(generated_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split=split,
            mode=mode,
        )
    )


def append_csv_row(csv_path: Path, row: Dict[str, Any]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_baseline_curve(csv_path: Path, output_png: Path):
    if not csv_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Skipping baseline curve plot.")
        return

    points = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = float(row["step"])
                fid = float(row["fid"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(step) and math.isfinite(fid):
                points.append((step, fid))

    if not points:
        return
    points.sort(key=lambda item: item[0])
    steps = [p[0] for p in points]
    fids = [p[1] for p in points]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, fids, marker="o", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("FID")
    plt.title("Baseline Curve: FID vs Step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    alpha: float,
    seed: int,
    use_ema: bool,
    dataset_split: str,
    fid_mode: str,
    cleanup_samples: bool,
) -> Dict[str, Any]:
    dataset_key = _dataset_key(dataset)
    if dataset_key != "cifar10":
        raise ValueError("This evaluator currently supports CIFAR-10 FID only (`--dataset cifar10`).")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step_metrics, epoch_metrics = _get_nested_metrics(checkpoint)

    step = int(checkpoint.get("step", -1))
    epoch = int(checkpoint.get("epoch", -1))
    artifact_dir = output_dir / "artifacts" / f"{checkpoint_path.stem}_step{step}"
    generated_dir = artifact_dir / "generated"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_info = build_model(checkpoint, dataset_key, device=device, use_ema=use_ema)

    start = time.time()
    generate_cifar10_images(
        model=model,
        out_dir=generated_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        in_channels=model_info["in_channels"],
        img_size=model_info["img_size"],
        num_classes=model_info["num_classes"],
        alpha=alpha,
        device=device,
        save_preview=True,
    )
    fid = compute_cifar10_fid(generated_dir, split=dataset_split, mode=fid_mode)
    elapsed = time.time() - start

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "dataset": dataset_key,
        "step": step,
        "epoch": epoch,
        "seed": seed,
        "torch_initial_seed": int(torch.initial_seed()),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "alpha": alpha,
        "use_ema": use_ema,
        "dataset_split": dataset_split,
        "fid_mode": fid_mode,
        "fid": fid,
        "eval_seconds": elapsed,
        "model_info": model_info,
    }
    with (artifact_dir / "eval_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if cleanup_samples:
        for image_path in generated_dir.glob("*.png"):
            image_path.unlink()
        generated_dir.rmdir()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "dataset": dataset_key,
        "epoch": epoch,
        "step": step,
        "fid": fid,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "alpha": alpha,
        "seed": seed,
        "loss": _metric(step_metrics, epoch_metrics, "loss", epoch_fallback="loss"),
        "drift_norm": _metric(step_metrics, epoch_metrics, "drift_norm", epoch_fallback="drift_norm"),
        "grad_norm": _metric(step_metrics, epoch_metrics, "grad_norm"),
        "feature_dist_mean": _metric(step_metrics, epoch_metrics, "feature_dist_mean"),
        "feature_dist_var": _metric(step_metrics, epoch_metrics, "feature_dist_var"),
        "avg_loss": _float_or_nan(epoch_metrics.get("loss")),
        "avg_drift_norm": _float_or_nan(epoch_metrics.get("drift_norm")),
        "eval_seconds": elapsed,
        "artifact_dir": str(artifact_dir.resolve()),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drifting Model checkpoints with clean-fid.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to one checkpoint (.pt)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of checkpoints")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar"], help="Dataset for FID stats")
    parser.add_argument("--output_dir", type=str, default="./eval", help="Evaluation output directory")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV (default: <output_dir>/fid_metrics.csv)")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of generated samples for FID")
    parser.add_argument("--batch_size", type=int, default=256, help="Generation batch size")
    parser.add_argument("--alpha", type=float, default=1.5, help="CFG alpha for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible evaluation")
    parser.add_argument("--no_ema", action="store_true", help="Use raw model weights instead of EMA")
    parser.add_argument("--dataset_split", type=str, default="train", choices=["train", "test"], help="Reference split for clean-fid")
    parser.add_argument("--fid_mode", type=str, default="clean", choices=["clean", "legacy_tensorflow", "legacy_pytorch"], help="clean-fid mode")
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoints already listed in CSV")
    parser.add_argument("--cleanup_samples", action="store_true", help="Delete generated image files after FID (keep metadata + preview)")
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args.checkpoint, args.checkpoint_dir)
    output_dir = Path(args.output_dir).resolve()
    csv_path = Path(args.csv_path).resolve() if args.csv_path else output_dir / "fid_metrics.csv"

    already_done = _read_existing_checkpoints(csv_path) if args.skip_existing else set()
    rows_written = 0

    for checkpoint_path in checkpoints:
        checkpoint_key = str(checkpoint_path.resolve())
        if checkpoint_key in already_done:
            print(f"Skipping already-evaluated checkpoint: {checkpoint_path}")
            continue

        print(f"Evaluating: {checkpoint_path}")
        row = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=args.dataset,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            alpha=args.alpha,
            seed=args.seed,
            use_ema=not args.no_ema,
            dataset_split=args.dataset_split,
            fid_mode=args.fid_mode,
            cleanup_samples=args.cleanup_samples,
        )
        append_csv_row(csv_path, row)
        rows_written += 1
        print(f"Step {row['step']} | FID {row['fid']:.4f} | wrote row to {csv_path}")

    save_baseline_curve(csv_path, output_dir / "fid_vs_step.png")
    print(f"Done. Rows written: {rows_written}")
    print(f"CSV: {csv_path}")
    print(f"Curve: {output_dir / 'fid_vs_step.png'}")


if __name__ == "__main__":
    main()
