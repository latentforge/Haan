"""Training execution layer.

  train.py     execution entry point (loop) — PyTorch train_loop + FSDP2 · PagedAdamW8bit · ckpt · logging
  runner.py    phase manager — per-phase config selection · transition · resume
  evaluate.py  standalone evaluation — checkpoint A/B/C · ASR WER/CER

Offline one-time tools live under project_amnesty/tools/ (derive_silence_codes.py,
inspect_moshi_weights.py), not here — they are not imported by the training loop.

[fixed] The location and names of train.py / runner.py must not change (even when
adding new paths later).
"""
