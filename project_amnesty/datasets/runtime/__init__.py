"""Runtime tier: unified Arrow -> collated batch. Runs every training step, as
pure `torch.utils.data` over the prepared Arrow.

Source-blind by construction: dataset.py branches on `sample_type` (the shape
contract), never on which corpus a row came from, so it may not import anything
from the offline tier. Within this tier, __getitem__ is a pure function of the
index; anything needing the batch or a model hyperparameter belongs to the
collator/sampler.
"""
