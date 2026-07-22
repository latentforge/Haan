"""Offline tier: raw corpora -> unified Arrow. Runs once per corpus (GPU-hours),
gated by prepare.py's sentinel.

These modules are the only ones that know where a sample came from -- per-corpus
parsing lives in sources/, the common encode/align/save pipeline in base.py +
mixins.py, and the Arrow write in prepare_dataset.py. The runtime tier never
imports anything here; that one-way dependency is the whole point of the split.

Import side effect: loading sources/ registers every builder in REGISTRY.
"""
