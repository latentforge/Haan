"""The data stack, end to end: raw corpora -> Arrow -> KDSample -> collated batch.

Both halves live here as one flat package. Importing it loads every source module,
which is what populates REGISTRY.

Offline (run once; writes data/prepared/{source}/{split}/):

    base.py             BaseDataset contract + auto-registry + AudioSourceDataset
    mixins.py           MimiEncoder / TextAlign / NpzPairIO
    *_dataset.py        one module per corpus (kss, zeroth_ko, en_kd, ...)
    schema.py           the unified Arrow schema + SCHEMA_VERSION
    prepare_dataset.py  all sources -> Arrow, per group
    prepare.py          ensure-materialized gate + `python -m project_amnesty.datasets.prepare`

Runtime (every step; pure `torch.utils.data` over the Arrow above):

    item.py      the per-sample contract
    config.py    TokenConfig / DataConfig
    crop.py      window selection (pure)
    dataset.py   MoshiKDDataset: Arrow -> KDSample
    collator.py  KDCollator: delay, padding, loss weights
    text_collator.py  TextAnchorCollator: the text-only group
    kd_align.py  teacher/student frame alignment (also the spec for losses/kd.py)
    sampler.py   MixingBatchSampler: group ratios, token budget, rank sharding
    schedule.py  MixSchedule: the Korean-ratio ramp
    loader.py    build_dataloader()

Two dividing lines survive the flattening, and both are load-bearing. The offline
modules are the only ones that know where a sample came from -- the runtime side
reads the unified Arrow and never sees `source`. And within the runtime side,
__getitem__ is a pure function of the index; anything that needs the batch or a
model hyperparameter belongs to the collator.

Caution: this must stay the `project_amnesty.datasets` subpackage. A top-level
`datasets/` would shadow the HF `datasets` import that base.py and dataset.py rely on.
"""

from .base import (
    REGISTRY,
    AudioSourceDataset,
    BaseDataset,
    RawEntry,
    build_dataset,
)
from .common_voice_ko_dataset import CommonVoiceKoDataset
from .config import DataConfig, TokenConfig
from .en_kd_dataset import EnKDDialogueDataset, FilterConfig
from .en_solo_dataset import EnSoloDataset
from .item import ITEM_KEYS, KDSample, validate_item
from .kss_dataset import KSSDataset
from .mixins import MimiEncoderMixin, NpzPairIOMixin, TextAlignMixin, TextTokCfg
from .seed_prompt_dataset import SeedPromptDataset
from .text_anchor_dataset import TextAnchorDataset
from .zeroth_ko_dataset import ZerothKoDataset

__all__ = [
    # offline
    "REGISTRY", "BaseDataset", "build_dataset",
    "MimiEncoderMixin", "NpzPairIOMixin", "TextAlignMixin", "TextTokCfg",
    "AudioSourceDataset", "RawEntry",
    "KSSDataset", "CommonVoiceKoDataset", "ZerothKoDataset",
    "EnKDDialogueDataset", "FilterConfig",
    "EnSoloDataset", "SeedPromptDataset", "TextAnchorDataset",
    # runtime
    "DataConfig", "TokenConfig", "KDSample", "ITEM_KEYS", "validate_item",
]
