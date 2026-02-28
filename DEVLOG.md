# DEVLOG

## Day 1 (2026-02-28)

### Summary
Day 1 focused on making training/evaluation reproducible and trackable:

1. Added pinned dependency manifests (`requirements.txt`, `pyproject.toml`) based on the active environment.
2. Added dataset YAML configs under `configs/` and wired training to load them.
3. Added deterministic debug controls for torch/cuda/cudnn seeding behavior.
4. Added structured experiment logging with Weights & Biases (W&B).
5. Added extra training diagnostics (`feature_dist_mean`, `feature_dist_var`) to logging.
6. Added checkpoint metric persistence so evaluations can read training stats.
7. Added a full checkpoint evaluation pipeline (`eval.py`) for CIFAR-10 FID over 50k generated samples.
8. Added CSV export per checkpoint and automatic baseline curve generation (FID vs step).

---

### What Changed

#### Dependency + environment reproducibility
- Added `requirements.txt` with pinned versions:
  - `torch==2.10.0+cu128`
  - `torchvision==0.25.0+cu128`
  - `einops==0.8.2`
  - `wandb`
  - `PyYAML`
  - `clean-fid`
- Added `pyproject.toml` with matching pinned dependencies and uv index config for CUDA 12.8 PyTorch wheels.

#### Config management
- Added:
  - `configs/mnist.yaml`
  - `configs/cifar10.yaml`
- Updated `train.py` to load YAML configs via `PyYAML`.
- Added CLI support to override config path and dataset root:
  - `--config`
  - `--data_root`

#### Deterministic debugging controls
- Updated `utils.set_seed()` to seed:
  - Python `random`
  - NumPy
  - Torch CPU
  - Torch CUDA (`manual_seed`, `manual_seed_all`)
- Added deterministic debug mode (opt-in):
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
- Added `train.py` flag:
  - `--deterministic_debug`

#### Structured training logging (W&B)
- Added a lightweight `StructuredLogger` in `train.py`.
- Added CLI flags:
  - `--logger {wandb,none}`
  - `--wandb_project`
  - `--wandb_run_name`
- Logged metrics each step:
  - `train/loss`
  - `train/drift_norm`
  - `train/grad_norm`
  - `train/feature_dist_mean`
  - `train/feature_dist_var`
  - `train/lr`

#### Feature-distance diagnostics
- Extended `drifting.compute_V()` with optional `return_dist_stats`.
- When enabled, it computes statistics over pairwise distances used for logits:
  - `feature_dist_mean`
  - `feature_dist_var`
- Integrated these into `compute_drifting_loss()` in `train.py` and propagated them through logging.

#### Checkpoint metadata for evaluation
- Extended `save_checkpoint()` (`utils.py`) to accept optional `metrics`.
- Training now saves latest step/epoch metrics into each checkpoint:
  - Step-level example: `loss`, `drift_norm`, `grad_norm`, `feature_dist_mean`, `feature_dist_var`, `lr`
  - Epoch-level example: `loss`, `drift_norm`, `num_batches`
- This enables evaluation CSV rows to include training context per checkpoint.

#### Documentation updates
- Updated `README.md`:
  - Install instructions now use `requirements.txt`
  - Added evaluation usage examples
  - Added `eval.py` to project structure

---

### `eval.py` Detailed Design

`eval.py` is a checkpoint-oriented evaluator for CIFAR-10. It computes quantitative generation quality (FID) and exports machine-readable tracking artifacts.

#### Primary purpose
For each checkpoint, produce:
1. A quantitative score (`fid`).
2. A durable artifact bundle (metadata + sample preview + optionally generated images).
3. A CSV row suitable for plotting and model selection.
4. An updated baseline chart (`FID vs step`).

#### Inputs
- One checkpoint: `--checkpoint path/to/checkpoint.pt`
- Or many checkpoints: `--checkpoint_dir path/to/dir`
- Required dataset mode: `--dataset cifar10` (or alias `cifar`)
- Evaluation controls:
  - `--num_samples` (default `50000`)
  - `--batch_size` (default `256`)
  - `--alpha` (CFG scale, default `1.5`)
  - `--seed` (default `42`)
  - `--dataset_split {train,test}` (default `train`)
  - `--fid_mode {clean,legacy_tensorflow,legacy_pytorch}` (default `clean`)
  - `--skip_existing`
  - `--cleanup_samples`

#### Execution flow per checkpoint
1. **Checkpoint load**
   - Reads model weights and checkpoint metadata (`step`, `epoch`, optional saved `metrics`).
2. **Model reconstruction**
   - Rebuilds the generator using stored config.
   - Uses EMA weights by default (`--no_ema` switches to raw model weights).
3. **Reproducibility setup**
   - Calls `set_seed(seed)` before sample generation.
   - Stores both configured seed and `torch_initial_seed` in artifact metadata.
4. **Sample generation (50k by default)**
   - Generates images in batches, clamped to `[-1, 1]`, converted to `uint8` PNGs.
   - Writes images into `eval/artifacts/<checkpoint>_step<step>/generated/`.
   - Saves `preview.png` for quick visual inspection.
5. **FID computation**
   - Uses `clean-fid` against CIFAR-10 reference stats:
     - `dataset_name="cifar10"`
     - `dataset_res=32`
     - split/mode from CLI flags
6. **Artifact metadata write**
   - Writes `eval_metadata.json` with run configuration, model info, seed data, FID, timing.
7. **CSV append**
   - Appends one row to `fid_metrics.csv` with checkpoint ID + quantitative metrics.
8. **Baseline curve refresh**
   - Rebuilds `fid_vs_step.png` from CSV (`step` on x-axis, `fid` on y-axis).

---

### What `eval.py` Measures

#### Core metric: FID
- **FID (Fréchet Inception Distance)** compares generated-image feature distribution to CIFAR-10 reference feature distribution.
- Lower is better.
- Intended to capture both sample quality and diversity mismatch.

#### Additional tracked values in CSV
Every checkpoint row includes:
- Checkpoint identity:
  - `checkpoint`, `checkpoint_name`, `dataset`, `epoch`, `step`
- Evaluation setup:
  - `num_samples`, `batch_size`, `alpha`, `seed`, `eval_seconds`
- Main score:
  - `fid`
- Training-context metrics (when present in checkpoint):
  - `loss`
  - `drift_norm`
  - `grad_norm`
  - `feature_dist_mean`
  - `feature_dist_var`
  - `avg_loss`
  - `avg_drift_norm`

Notes:
- Older checkpoints that do not include saved metric metadata will have `NaN` for some of these context columns.
- KID is not implemented yet in this script (FID-only at present).

---

### Files Added / Updated

Added:
- `DEVLOG.md`
- `eval.py`
- `configs/mnist.yaml`
- `configs/cifar10.yaml`
- `requirements.txt`
- `pyproject.toml`

Updated:
- `train.py`
- `drifting.py`
- `utils.py`
- `README.md`

---

### Example commands

Training (CIFAR-10):
```bash
python train.py --dataset cifar10 --logger wandb --wandb_project drifting-model
```

Evaluate one checkpoint:
```bash
python eval.py --checkpoint outputs/cifar10/checkpoint_final.pt --dataset cifar10
```

Evaluate a checkpoint directory and build baseline curve:
```bash
python eval.py --checkpoint_dir outputs/cifar10 --dataset cifar10 --skip_existing
```
