"""One module per corpus builder. Importing this package registers every builder.

Registration is an import side effect (BaseDataset.__init_subclass__ fills
REGISTRY), so a builder module that is not imported here silently disappears
from the CLI and from prepare -- no error, the corpus just stops existing.
Every new <name>_dataset.py MUST be imported below.
"""

from .common_voice_ko_dataset import CommonVoiceKoDataset
from .en_kd_dataset import EnKDDialogueDataset, FilterConfig
from .en_solo_dataset import EnSoloDataset
from .kss_dataset import KSSDataset
from .seed_prompt_dataset import SeedPromptDataset
from .text_anchor_dataset import TextAnchorDataset
from .zeroth_ko_dataset import ZerothKoDataset

__all__ = [
    "CommonVoiceKoDataset",
    "EnKDDialogueDataset", "FilterConfig",
    "EnSoloDataset",
    "KSSDataset",
    "SeedPromptDataset",
    "TextAnchorDataset",
    "ZerothKoDataset",
]
