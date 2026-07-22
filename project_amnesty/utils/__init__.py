"""Training execution layer.

  train.py     execution entry point (loop) — PyTorch train_loop + FSDP2 · PagedAdamW8bit · ckpt · logging
  runner.py    phase manager — per-phase config selection · transition · resume
  evaluate.py  standalone evaluation — checkpoint A/B/C · ASR WER/CER
  warm_start_haan.py  Moshi -> Haan weight transfer (ARCH 5.4.1/5.4.2) — a training-time
               conversion, kept here rather than in models/ so the dependency stays one-way
               (utils -> models). Reads a *Moshi* checkpoint; `from_pretrained` reads a Haan one.

Offline one-time tools live under project_amnesty/tools/ (derive_silence_codes.py,
inspect_moshi_weights.py), not here — they are not imported by the training loop.

[fixed] The location and names of train.py / runner.py must not change (even when
adding new paths later).
"""
