"""Haan — stock Moshi → PersonaPlex reproduction.

Qwen3-8B backbone (Temporal) + audio RVQ tokens + shared self/user embeddings +
Role Token + a shared Depth Transformer (batch-2 parallel). The goal is to test
whether full-duplex multi-turn behavior emerges from Korean single-turn data alone.

All training code lives inside this package.
  models/    HF-Moshi-mirror naming (configuration_/modeling_/processing_haan.py) — later
  datasets/  data loading & preprocessing (absolute imports; distinct from HF `datasets`)
  utils/     training entry point, phase manager, evaluation, tools

Design references: docs/contexts/{ARCHITECTURE, DATA_STRATEGY, RISKS_AND_DIAGNOSTICS,
TRAINING_CURRICULUM}.md — the "§" references in these files point to those documents.
"""
