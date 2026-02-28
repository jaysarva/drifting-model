"""
Training script for Drifting Models on MNIST and CIFAR-10.
Implements Algorithm 1 from the paper with class-conditional generation.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DriftDiT_models
from drifting import compute_V
from feature_encoder import create_feature_encoder
from utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)


# Default hyperparameters
MNIST_CONFIG = {
    "model": "DriftDiT-Tiny",
    "img_size": 32,
    "in_channels": 1,
    "num_classes": 10,
    "batch_nc": 10,  # Number of classes per batch
    "batch_n_pos": 32,  # Positive samples per class
    "batch_n_neg": 32,  # Negative samples per class
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "warmup_steps": 1000,
    "epochs": 100,
    "alpha_min": 1.0,
    "alpha_max": 3.0,
    "use_feature_encoder": False,  # Pixel space for MNIST
    "queue_size": 128,
    "label_dropout": 0.1,
}

CIFAR10_CONFIG = {
    "model": "DriftDiT-Small",
    "img_size": 32,
    "in_channels": 3,
    "num_classes": 10,
    "batch_nc": 10,
    "batch_n_pos": 32,
    "batch_n_neg": 32,
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "warmup_steps": 2000,
    "epochs": 200,
    "alpha_min": 1.0,
    "alpha_max": 3.0,
    "use_feature_encoder": True,  # Use pretrained ResNet for feature space
    "queue_size": 128,
    "label_dropout": 0.1,
}


DEFAULT_DATA_ROOT = "/home/qingtianzhu.ty/drifting/data"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def load_training_config(dataset: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset config from defaults + YAML overrides."""
    dataset_key = dataset.lower()
    if dataset_key == "cifar":
        dataset_key = "cifar10"
    if dataset_key not in {"mnist", "cifar10"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    config = (MNIST_CONFIG if dataset_key == "mnist" else CIFAR10_CONFIG).copy()
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parent / "configs" / f"{dataset_key}.yaml"
    )

    if path.exists():
        config.update(load_yaml_config(path))
    elif config_path is not None:
        raise FileNotFoundError(f"Config path does not exist: {path}")

    return config


class StructuredLogger:
    """Structured metric logging backend (W&B)."""

    def __init__(
        self,
        backend: str,
        output_dir: Path,
        config: Dict[str, Any],
        dataset: str,
        project: str,
        run_name: Optional[str] = None,
    ):
        self.backend = backend
        self._wandb = None
        self.run = None

        if backend == "none":
            return
        if backend != "wandb":
            raise ValueError(f"Unknown logger backend: {backend}")

        try:
            import wandb
        except ImportError:
            print("wandb is not installed. Structured logging is disabled.")
            self.backend = "none"
            return

        resolved_run_name = run_name or f"{dataset}-{int(time.time())}"
        self._wandb = wandb
        try:
            self.run = wandb.init(
                project=project,
                name=resolved_run_name,
                config=config,
                dir=str(output_dir),
                reinit=True,
            )
        except Exception as exc:
            print(f"wandb init failed ({exc}). Structured logging is disabled.")
            self.backend = "none"
            self.run = None

    def log(self, metrics: Dict[str, float], step: int):
        if self.run is None:
            return
        self._wandb.log(metrics, step=step)

    def close(self):
        if self.run is not None:
            self._wandb.finish()
            self.run = None


def get_dataset(name: str, root: str = DEFAULT_DATA_ROOT) -> tuple:
    """Get dataset and transforms."""
    if name.lower() == "mnist":
        # MNIST data is at {root}/mnist/MNIST/raw/
        mnist_root = os.path.join(root, "mnist")
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
        train_dataset = datasets.MNIST(mnist_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(mnist_root, train=False, download=True, transform=transform)
    elif name.lower() in ["cifar10", "cifar"]:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset


def sample_batch(
    queue: SampleQueue,
    num_classes: int,
    n_pos: int,
    device: torch.device,
) -> tuple:
    """Sample a batch of positive samples from the queue."""
    x_pos_list = []
    labels_list = []

    for c in range(num_classes):
        x_c = queue.sample(c, n_pos, device)
        x_pos_list.append(x_c)
        labels_list.append(torch.full((n_pos,), c, device=device, dtype=torch.long))

    x_pos = torch.cat(x_pos_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return x_pos, labels


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    use_pixel_space: bool = False,
) -> tuple:
    """
    Compute class-conditional drifting loss with multi-scale features.

    Following paper Section A.5: compute drifting loss at each scale, then sum.

    Args:
        x_gen: Generated samples (B, C, H, W)
        labels_gen: Labels for generated samples (B,)
        x_pos: Positive (real) samples (B_pos, C, H, W)
        labels_pos: Labels for positive samples (B_pos,)
        feature_encoder: Feature encoder (returns List[Tensor] for multi-scale)
        temperatures: List of temperatures for V computation
        use_pixel_space: Whether to use pixel space directly

    Returns:
        loss: Scalar loss
        info: Dict with metrics
    """
    device = x_gen.device
    num_classes = labels_gen.max().item() + 1

    # Extract features
    if use_pixel_space or feature_encoder is None:
        # Pixel space: single scale
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        # Multi-scale feature maps from pretrained encoder
        feat_gen_maps = feature_encoder(x_gen)  # List of (B, C, H, W)
        with torch.no_grad():
            feat_pos_maps = feature_encoder(x_pos)

        # Global average pool each scale to get vectors
        feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
        feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_maps]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    total_feature_dist_mean = 0.0
    total_feature_dist_var = 0.0
    num_losses = 0

    # Compute loss per class
    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c

        if not mask_gen.any() or not mask_pos.any():
            continue

        # Compute loss at each scale
        for feat_gen, feat_pos in zip(feat_gen_list, feat_pos_list):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]

            # Negatives: generated samples from current class (following Algorithm 1: y_neg = x)
            feat_neg_c = feat_gen_c

            # Simple L2 normalization (projects to unit sphere)
            feat_gen_c_norm = F.normalize(feat_gen_c, p=2, dim=1)
            feat_pos_c_norm = F.normalize(feat_pos_c, p=2, dim=1)
            feat_neg_c_norm = F.normalize(feat_neg_c, p=2, dim=1)

            # Compute V with multiple temperatures
            V_total = torch.zeros_like(feat_gen_c_norm)
            dist_stats = None
            for tau_idx, tau in enumerate(temperatures):
                if tau_idx == 0:
                    V_tau, dist_stats = compute_V(
                        feat_gen_c_norm,
                        feat_pos_c_norm,
                        feat_neg_c_norm,
                        tau,
                        mask_self=True,  # y_neg = x, so mask self
                        return_dist_stats=True,
                    )
                else:
                    V_tau = compute_V(
                        feat_gen_c_norm,
                        feat_pos_c_norm,
                        feat_neg_c_norm,
                        tau,
                        mask_self=True,  # y_neg = x, so mask self
                    )
                # Normalize each V before summing
                v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V_tau = V_tau / (v_norm + 1e-8)
                V_total = V_total + V_tau

            # Loss: MSE(phi(x), stopgrad(phi(x) + V))
            target = (feat_gen_c_norm + V_total).detach()
            loss_scale = F.mse_loss(feat_gen_c_norm, target)

            total_loss = total_loss + loss_scale
            total_drift_norm += (V_total ** 2).mean().item() ** 0.5
            if dist_stats is not None:
                total_feature_dist_mean += dist_stats["feature_dist_mean"]
                total_feature_dist_var += dist_stats["feature_dist_var"]
            num_losses += 1

    if num_losses == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0,
            "drift_norm": 0.0,
            "feature_dist_mean": 0.0,
            "feature_dist_var": 0.0,
        }

    loss = total_loss / num_losses
    info = {
        "loss": loss.item(),
        "drift_norm": total_drift_norm / num_losses,
        "feature_dist_mean": total_feature_dist_mean / num_losses,
        "feature_dist_var": total_feature_dist_var / num_losses,
    }

    return loss, info


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    feature_encoder: Optional[nn.Module] = None,
) -> dict:
    """
    Single training step (Algorithm 1).

    1. Sample class labels and CFG alpha
    2. Generate samples from noise
    3. Sample positive samples from queue
    4. Compute drifting field and loss
    5. Update model
    """
    model.train()
    num_classes = config["num_classes"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    alpha_min = config["alpha_min"]
    alpha_max = config["alpha_max"]
    temperatures = config["temperatures"]
    use_pixel = not config["use_feature_encoder"]

    # Total batch size
    batch_size = num_classes * n_neg

    # Sample class labels (repeat each class n_neg times)
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)

    # Sample CFG alpha ~ Uniform(alpha_min, alpha_max)
    alpha = torch.empty(batch_size, device=device).uniform_(alpha_min, alpha_max)

    # Sample noise
    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    # Generate samples
    x_gen = model(noise, labels, alpha)

    # Sample positive samples from queue
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)

    # Compute drifting loss
    loss, info = compute_drifting_loss(
        x_gen,
        labels,
        x_pos,
        labels_pos,
        feature_encoder,
        temperatures,
        use_pixel_space=use_pixel,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["grad_clip"]
    )
    info["grad_norm"] = grad_norm.item()

    # Optimizer step
    optimizer.step()

    return info


def fill_queue(
    queue: SampleQueue,
    dataloader: DataLoader,
    device: torch.device,
    min_samples: int = 64,
):
    """Fill the sample queue with real data."""
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, labels = batch[0], batch[1]
        else:
            x, labels = batch, torch.zeros(batch.shape[0], dtype=torch.long)

        queue.add(x, labels)

        if queue.is_ready(min_samples):
            break


def train(
    dataset: str = "mnist",
    output_dir: str = "./outputs",
    config_path: Optional[str] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    resume: Optional[str] = None,
    seed: int = 42,
    deterministic_debug: bool = False,
    num_workers: int = 4,
    logger_backend: str = "wandb",
    wandb_project: str = "drifting-model",
    wandb_run_name: Optional[str] = None,
    log_interval: int = 100,
    save_interval: int = 10,
    sample_interval: int = 10,
):
    """Main training function."""
    set_seed(seed, deterministic_debug=deterministic_debug)
    if deterministic_debug:
        print("Deterministic debug mode enabled: CuDNN deterministic=True, benchmark=False")

    # Get config
    config = load_training_config(dataset, config_path=config_path)
    config["dataset"] = dataset
    config["seed"] = seed
    config["deterministic_debug"] = deterministic_debug

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = StructuredLogger(
        backend=logger_backend,
        output_dir=output_dir,
        config=config,
        dataset=dataset,
        project=wandb_project,
        run_name=wandb_run_name,
    )

    # Load dataset
    train_dataset, _ = get_dataset(dataset, root=data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
    ).to(device)

    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    # Create EMA
    ema = EMA(model, decay=config["ema_decay"])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )

    # Create scheduler
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )

    # Create sample queue
    queue = SampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    # Feature encoder (for CIFAR)
    feature_encoder = None
    if config["use_feature_encoder"]:
        print("Creating feature encoder...")
        feature_encoder = create_feature_encoder(
            dataset=dataset,
            feature_dim=512,
            multi_scale=True,
            use_pretrained=True,  # Use ImageNet-pretrained ResNet
        ).to(device)

        # For pretrained ResNet, no need for MAE pre-training
        # The ImageNet features work well for natural images
        print("Using ImageNet-pretrained ResNet encoder")

        feature_encoder.eval()
        for param in feature_encoder.parameters():
            param.requires_grad = False

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    last_step_metrics: Dict[str, float] = {}
    last_epoch_metrics: Dict[str, float] = {}
    if resume:
        checkpoint = load_checkpoint(resume, model, ema, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        if isinstance(checkpoint.get("metrics"), dict):
            metrics = checkpoint["metrics"]
            if isinstance(metrics.get("step"), dict):
                last_step_metrics = dict(metrics["step"])
            if isinstance(metrics.get("epoch"), dict):
                last_epoch_metrics = dict(metrics["epoch"])
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        # Fill queue at start of each epoch
        fill_queue(queue, train_loader, device, min_samples=64)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                x_real, labels_real = batch[0].to(device), batch[1].to(device)
            else:
                x_real = batch.to(device)
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=device)

            # Add to queue
            queue.add(x_real.cpu(), labels_real.cpu())

            # Skip if queue not ready
            if not queue.is_ready(config["batch_n_pos"]):
                continue

            # Training step
            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                feature_encoder,
            )

            # Update EMA and scheduler
            ema.update(model)
            scheduler.step()

            # Accumulate metrics
            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1
            lr = scheduler.get_lr()

            logger.log(
                {
                    "train/loss": info["loss"],
                    "train/drift_norm": info["drift_norm"],
                    "train/grad_norm": info["grad_norm"],
                    "train/feature_dist_mean": info["feature_dist_mean"],
                    "train/feature_dist_var": info["feature_dist_var"],
                    "train/lr": lr,
                },
                step=global_step,
            )
            last_step_metrics = {
                "loss": info["loss"],
                "drift_norm": info["drift_norm"],
                "grad_norm": info["grad_norm"],
                "feature_dist_mean": info["feature_dist_mean"],
                "feature_dist_var": info["feature_dist_var"],
                "lr": lr,
            }

            # Logging
            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Drift: {info['drift_norm']:.4f} | "
                    f"Grad: {info['grad_norm']:.4f} | "
                    f"DistMean: {info['feature_dist_mean']:.4f} | "
                    f"DistVar: {info['feature_dist_var']:.4f} | "
                    f"LR: {lr:.6f}"
                )

            # Generate samples every 500 steps for quick visualization
            if global_step % 500 == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                generate_samples(
                    ema.shadow,
                    config,
                    device,
                    str(sample_path),
                    num_per_class=8,
                )
                print(f"Saved samples to {sample_path}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        last_epoch_metrics = {
            "loss": avg_loss,
            "drift_norm": avg_drift,
            "num_batches": float(num_batches),
        }
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Drift Norm: {avg_drift:.4f}\n"
        )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                metrics={
                    "step": last_step_metrics,
                    "epoch": last_epoch_metrics,
                },
            )
            print(f"Saved checkpoint to {ckpt_path}")

        # Generate samples
        if (epoch + 1) % sample_interval == 0:
            sample_path = output_dir / f"samples_epoch{epoch+1}.png"
            generate_samples(
                ema.shadow,
                config,
                device,
                str(sample_path),
                num_per_class=8,
            )
            print(f"Saved samples to {sample_path}")

    # Final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        ema,
        optimizer,
        scheduler,
        config["epochs"] - 1,
        global_step,
        config,
        metrics={
            "step": last_step_metrics,
            "epoch": last_epoch_metrics,
        },
    )
    logger.close()
    print(f"Training complete! Final checkpoint saved to {final_path}")


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    config: dict,
    device: torch.device,
    save_path: str,
    num_per_class: int = 8,
    alpha: float = 1.5,
):
    """Generate samples for visualization."""
    model.eval()

    num_classes = config["num_classes"]
    in_channels = config["in_channels"]
    img_size = config["img_size"]

    # Generate samples for each class
    samples = []
    for c in range(num_classes):
        noise = torch.randn(num_per_class, in_channels, img_size, img_size, device=device)
        labels = torch.full((num_per_class,), c, device=device, dtype=torch.long)
        alpha_tensor = torch.full((num_per_class,), alpha, device=device)

        # Use CFG
        x = model.forward_with_cfg(noise, labels, alpha=alpha)
        samples.append(x)

    samples = torch.cat(samples, dim=0)
    samples = samples.clamp(-1, 1)

    save_image_grid(samples, save_path, nrow=num_per_class)


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides dataset defaults)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--deterministic_debug",
        action="store_true",
        help="Enable deterministic torch/cudnn behavior for debugging (slower).",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "none"],
        help="Structured logger backend",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="drifting-model",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Checkpoint save interval (epochs)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Sample generation interval (epochs)",
    )

    args = parser.parse_args()

    train(
        dataset=args.dataset,
        output_dir=args.output_dir,
        config_path=args.config,
        data_root=args.data_root,
        resume=args.resume,
        seed=args.seed,
        deterministic_debug=args.deterministic_debug,
        num_workers=args.num_workers,
        logger_backend=args.logger,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
    )


if __name__ == "__main__":
    main()
