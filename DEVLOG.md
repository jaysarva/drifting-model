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

## Day 2 (2026-02-28)

### Summary
Day 2 implemented **Learned Drifting Fields (LDF v0)** as a configurable drop-in replacement for the hand-designed kernel field, while preserving the core training invariant:

- antisymmetric field construction
- fixed-point stopgrad target (`target = (feat + V).detach()`)

This adds a direct path to test whether learned set geometry can reduce dependence on encoder strength.

---

### What Changed

#### 1. Added learned antisymmetric set operator (`learned_field.py`)

Implemented:
- `SetAttentionField`
- `LearnedFieldBank`
- `compute_V_learned(...)` functional wrapper

Core construction:
- `V(x; P, Q) = S(x, P) - S(x, Q)` (antisymmetric by design)

Set operator details:
- Optional projection `g = MLP(feat)` (toggleable)
- Attention logits from:
  - `[g(x), g(y), g(y)-g(x), ||g(y)-g(x)||]`
- Softmax over the set
- Output:
  - weighted sum of `(g(y)-g(x))`
  - projected back to feature dimension when projection is enabled

Also added safe handling for singleton self-masked negatives to avoid invalid softmax states.

#### 2. Training pipeline integration (`train.py`)

Added config-controlled switch:
- `field_type: kernel | learned`

Added LDF-related config knobs:
- `kernel_use_multi_temperature`
- `ldf_hidden_dim`
- `ldf_projection_dim`
- `ldf_score_hidden_dim`
- `ldf_use_projection`
- `ldf_mask_self_neg`

Behavioral changes:
- `compute_drifting_loss(...)` now branches between:
  - original kernel field
  - learned field (`learned_field.compute_V_learned(...)`)
- LDF path keeps the same stopgrad target as baseline.
- Kernel multi-temperature remains available for baseline.
- When `field_type=learned`, multi-temperature kernel mixing is disabled in config to avoid confounds.

Optimization changes:
- LDF parameters are added to the optimizer parameter list.
- Gradient clipping now includes both model + LDF params.

#### 3. Checkpoint compatibility (`utils.py`, `train.py`)

Extended checkpoint saving with optional `extra_state`.
Used to persist:
- `learned_field` state dict (only when LDF is active)

Resume behavior:
- If training with LDF and checkpoint contains `learned_field`, it is restored.
- If missing, training warns and continues with freshly initialized LDF weights.

#### 4. Config and ablation setup (`configs/`)

Updated baseline configs to include field/LDF keys:
- `configs/mnist.yaml`
- `configs/cifar10.yaml`
- `configs/cifar10-short.yaml`

Added dedicated LDF ablation configs:
- `configs/cifar10-ldf.yaml` (learned attention + projection)
- `configs/cifar10-ldf-no-proj.yaml` (without `g()` projection)
- `configs/cifar10-ldf-setsize-small.yaml` (`Npos=Nneg=16`)
- `configs/cifar10-ldf-setsize-large.yaml` (`Npos=Nneg=64`)

Aligned these LDF configs to the existing short-baseline budget for fair comparison with `configs/cifar10-short.yaml`:
- `epochs: 50`
- `warmup_steps: 500`

These directly cover requested ablations:
- learned attention vs fixed kernel
- with/without projection
- set size sensitivity

#### 5. Tests for antisymmetry and basic correctness (`tests/test_learned_field.py`)

Added tests for:
- antisymmetry with projection
- antisymmetry without projection
- finite output for singleton self-masked negatives
- multi-scale bank output shape sanity

#### 6. Documentation update (`README.md`)

Added:
- LDF mention in method summary
- training command for LDF config
- ablation command examples

#### 7. Test runner compatibility fix (`__init__.py`)

Fixed `pytest` collection failure caused by importing repo-root `__init__.py` as a top-level module.

Change:
- Added dual import path support in `__init__.py`:
  - package-relative imports when imported as a package
  - local absolute imports fallback for top-level import contexts

Result:
- tests now run successfully inside `/.venv`.

#### 8. Automated LDF verification script (`scripts/run_ldf_verification.sh`)

Added an end-to-end script that runs the short-budget LDF comparison matrix (single seed, matching baseline seed 42):

- `ldf_short` (learned attention + projection)
- `ldf_no_proj_short` (projection ablation)
- `ldf_set_small_short` (`Npos/Nneg=16`)
- `ldf_set_large_short` (`Npos/Nneg=64`)

Design choices in script:
- Activates `/.venv` at start.
- Uses fixed logging backend:
  - `--logger wandb`
  - `--wandb_project drifting-model`
- Writes eval outputs to run-specific directories under:
  - `eval/ldf_verification_<RUN_TAG>/...`
- Cleans disk usage after each experiment:
  - deletes `<eval_run>/artifacts/`
  - deletes corresponding training output directory under `outputs/...`
- Assumes kernel short baseline already exists and therefore does **not** rerun kernel.

#### 9. Short CIFAR run without feature encoder (`configs/cifar10-short-no-encoder.yaml`, `scripts/run_cifar10_short_no_encoder.sh`)

Added a short-budget kernel config that matches the prior CIFAR short baseline except:
- `use_feature_encoder: false`

Also added a dedicated train+eval script to run this comparison first:
- `scripts/run_cifar10_short_no_encoder.sh`

Script behavior:
- Activates `/.venv`
- Trains CIFAR-10 short run with no feature encoder (seed 42 by default)
- Evaluates with `eval.py` into run-specific `eval/...` directory
- Deletes eval `artifacts/` to reduce disk usage
- Keeps training outputs/checkpoints for inspection and potential resume

---

### Baseline Eval Analysis (`configs/cifar10-short.yaml`, produced with `eval.py`)

Evaluated checkpoints (EMA, clean-fid, CIFAR-10 train split, 50k generated samples, `alpha=1.5`, `seed=42`):

- step 1950 (epoch 10): FID **155.97**
- step 3900 (epoch 20): FID **102.59**
- step 5850 (epoch 30): FID **97.63**
- step 7800 (epoch 40): FID **88.67**
- step 9750 (epoch 50): FID **79.92**

Curve behavior (`eval_cifar10-short/fid_vs_step.png`):
- Monotonic decrease across all measured checkpoints.
- Very large early gain from 1950 -> 3900 (about **-53.38 FID**).
- After that, improvements continue but at a slower rate (roughly **-5 to -9 FID** every 1950 steps).
- Net improvement over short run: **-76.05 FID** (about **48.8%** reduction vs step 1950).

Additional notes from `eval_cifar10-short/fid_metrics.csv`:
- `checkpoint_epoch50.pt` and `checkpoint_final.pt` have identical `step=9750` and identical FID (**79.92**), which is expected since they represent the same terminal training state.
- Training-side metrics (`loss`, `drift_norm`) change only modestly while FID improves substantially, indicating those internal statistics are not strongly predictive of sample quality in this run.

Interpretation:
- The baseline is clearly learning in the short schedule and has not plateaued hard by step 9750.
- The final short-run FID (~80) is still far from strong CIFAR baselines, so there is headroom for architectural/operator improvements (including LDF) and/or longer training.

---

### Validation Performed

- `python -m compileall .` passed.
- In `/.venv`, `pytest -q tests/test_learned_field.py` passed (`4 passed`).
- In `/.venv`, `pytest -q` passed (`4 passed`).

---

### Files Added / Updated

Added:
- `learned_field.py`
- `tests/test_learned_field.py`
- `configs/cifar10-ldf.yaml`
- `configs/cifar10-ldf-no-proj.yaml`
- `configs/cifar10-ldf-setsize-small.yaml`
- `configs/cifar10-ldf-setsize-large.yaml`
- `configs/cifar10-short-no-encoder.yaml`
- `scripts/run_ldf_verification.sh`
- `scripts/run_cifar10_short_no_encoder.sh`

Updated:
- `train.py`
- `utils.py`
- `README.md`
- `__init__.py`
- `configs/mnist.yaml`
- `configs/cifar10.yaml`
- `configs/cifar10-short.yaml`

---

### Next Measurement Step

1. Run the short no-feature-encoder comparison first (same budget as existing baseline):

```bash
bash scripts/run_cifar10_short_no_encoder.sh
```

Goal:
- Decide whether short-budget CIFAR training is viable without a feature encoder in this implementation.
- Compare this run’s FID curve directly against existing `configs/cifar10-short.yaml` baseline (`use_feature_encoder: true`).

2. Then run LDF ablations:

```bash
bash scripts/run_ldf_verification.sh
```

This second script tests/ablates (single seed, short budget):
- Learned field vs existing short kernel baseline (baseline already produced separately)
- With projection vs without projection
- Set-size sensitivity (`Npos/Nneg=16` vs `64`)
