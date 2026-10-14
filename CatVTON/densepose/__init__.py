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

try:
    from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
    from .modeling.roi_heads import DensePoseROIHeads
    from .modeling.test_time_augmentation import (
        DensePoseGeneralizedRCNNWithTTA,
        DensePoseDatasetMapperTTA,
    )
    from .utils.transform import load_from_cfg
    from .modeling.hrfpn import build_hrfpn_backbone
    from .data.datasets import builtin
    from .converters import builtin as builtin_converters
    from .evaluation import DensePoseCOCOEvaluator
except Exception:
    pass


