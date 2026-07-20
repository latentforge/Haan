"""Offline / one-time diagnostic tools (not part of the training loop).

  derive_silence_codes.py   one-time data-prep: derive the Mimi silence-code bank -> configs/data/mimi_silence.json
  inspect_moshi_weights.py  weight audit: compare the original Moshi self/user audio-embedding tables
                            (reproduces ARCHITECTURE.md 3.5.1 -- row cosine, constant-offset share, residual share)

These are run by hand (`python -m project_amnesty.tools.<name>`), not imported by
train.py / runner.py. Heavy deps (torch, transformers, huggingface_hub) are imported
lazily inside the functions so the package stays importable without them.
"""
