# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
from .config import (
    add_densepose_config,
    add_densepose_head_config,
    add_hrnet_config,
    add_dataset_category_config,
    add_bootstrap_config,
    load_bootstrap_config,
)

# Explicitly register modeling components with Detectron2
try:
    from .modeling.roi_heads import DensePoseROIHeads
except Exception:
    pass

try:
    from .modeling.hrfpn import build_hrfpn_backbone
except Exception:
    pass

try:
    from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
except Exception:
    pass

try:
    from .modeling.test_time_augmentation import (
        DensePoseGeneralizedRCNNWithTTA,
        DensePoseDatasetMapperTTA,
    )
except Exception:
    pass

try:
    from .utils.transform import load_from_cfg
except Exception:
    pass

try:
    from .data.datasets import builtin
except Exception:
    pass

try:
    from .converters import builtin as builtin_converters
except Exception:
    pass

try:
    from .evaluation import DensePoseCOCOEvaluator
except Exception:
    pass


