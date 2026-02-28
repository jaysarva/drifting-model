"""
Drifting Models Implementation

A PyTorch reproduction of "Generative Modeling via Drifting" (Deng et al., 2026)
for MNIST and CIFAR-10 datasets.
"""

if __package__:
    from .model import DriftDiT, DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
    from .drifting import (
        compute_V,
        compute_V_multi_temperature,
        normalize_features,
        normalize_drift,
        DriftingLoss,
        ClassConditionalDriftingLoss,
        drift_step_2d,
    )
    from .feature_encoder import (
        MultiScaleFeatureEncoder,
        MAEEncoder,
        create_feature_encoder,
        pretrain_mae,
    )
    from .learned_field import (
        SetAttentionField,
        LearnedFieldBank,
        compute_V_learned,
    )
    from .utils import (
        EMA,
        WarmupLRScheduler,
        SampleQueue,
        save_checkpoint,
        load_checkpoint,
        save_image_grid,
        visualize_samples,
        count_parameters,
        set_seed,
    )
else:
    # Support direct top-level import (e.g., pytest collection of repository root __init__.py).
    from model import DriftDiT, DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
    from drifting import (
        compute_V,
        compute_V_multi_temperature,
        normalize_features,
        normalize_drift,
        DriftingLoss,
        ClassConditionalDriftingLoss,
        drift_step_2d,
    )
    from feature_encoder import (
        MultiScaleFeatureEncoder,
        MAEEncoder,
        create_feature_encoder,
        pretrain_mae,
    )
    from learned_field import (
        SetAttentionField,
        LearnedFieldBank,
        compute_V_learned,
    )
    from utils import (
        EMA,
        WarmupLRScheduler,
        SampleQueue,
        save_checkpoint,
        load_checkpoint,
        save_image_grid,
        visualize_samples,
        count_parameters,
        set_seed,
    )

__version__ = "0.1.0"
__all__ = [
    # Model
    "DriftDiT",
    "DriftDiT_Tiny",
    "DriftDiT_Small",
    "DriftDiT_models",
    # Drifting
    "compute_V",
    "compute_V_multi_temperature",
    "normalize_features",
    "normalize_drift",
    "DriftingLoss",
    "ClassConditionalDriftingLoss",
    "drift_step_2d",
    "SetAttentionField",
    "LearnedFieldBank",
    "compute_V_learned",
    # Feature encoder
    "MultiScaleFeatureEncoder",
    "MAEEncoder",
    "create_feature_encoder",
    "pretrain_mae",
    # Utils
    "EMA",
    "WarmupLRScheduler",
    "SampleQueue",
    "save_checkpoint",
    "load_checkpoint",
    "save_image_grid",
    "visualize_samples",
    "count_parameters",
    "set_seed",
]
