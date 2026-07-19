"""Training-time data stack: prepared Arrow -> KDSample -> collated batch.

Layering (see docs/plans/MOSHI_KD_DATASET_PLAN.md):

    item.py      the per-sample contract
    config.py    TokenConfig / DataConfig
    crop.py      window selection (pure)
    dataset.py   MoshiKDDataset: Arrow -> KDSample
    collator.py  KDCollator: delay, padding, loss weights
    sampler.py   MixingBatchSampler: group ratios, token budget, rank sharding
    schedule.py  MixSchedule: the Korean-ratio ramp
    loader.py    build_dataloader()

The dividing line: __getitem__ is a pure function of the index; anything that
needs the batch or a model hyperparameter belongs to the collator.
"""

from .config import DataConfig, TokenConfig
from .item import ITEM_KEYS, KDSample, validate_item

__all__ = ["DataConfig", "TokenConfig", "KDSample", "ITEM_KEYS", "validate_item"]
