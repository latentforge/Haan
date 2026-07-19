"""CLI: python -m data_pipeline.datasets <name> [options]

Examples:
  # KO TTS sources (audio family)
  python -m data_pipeline.datasets kss --root data/raw/kss --out-dir data/tokenized
  python -m data_pipeline.datasets zeroth_ko --root . --out-dir data/tokenized --limit 100

  # en_kd: ingest teacher self-play dialogues, then filter. Generation itself is
  # NOT done here -- run the Moshi self-play harness to get dialogue_*.npz first.
  python -m data_pipeline.datasets en_kd --stage ingest --root <dialogues_dir> \
      --text-config configs/data/text_tok.yaml --filter-config configs/data/filter.yaml
  python -m data_pipeline.datasets en_kd --stage filter \
      --filter-config configs/data/filter.yaml            # threshold tuning: filter only

  # en_solo: crops from en_kd artifacts (--root = en_kd artifact directory)
  python -m data_pipeline.datasets en_solo --root data/generated/en_kd \
      --out-dir data/generated --text-config configs/data/text_tok.yaml

  python -m data_pipeline.datasets --list
"""

import argparse

if __package__ in (None, ""):
    # Run directly as a script (e.g. VSCode ▶): no package context, so relative
    # imports fail. Put the repo root on sys.path and import absolutely.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_pipeline.datasets import REGISTRY
else:
    from . import REGISTRY


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", help=f"dataset name: {sorted(REGISTRY)}")
    p.add_argument("--root", default=".", help="raw data path (for en_solo: en_kd artifact path)")
    # No default: each builder declares its own in __init__ (tokenized / generated /
    # data), and from_cli falls back to it. A shared default here would silently
    # override those -- e.g. seed_prompts landing in data/tokenized/seed_prompts/
    # while its consumer reads data/seed_prompts/, with no error either side.
    p.add_argument("--out-dir", default=None, help="artifact root (actual location: <out-dir>/<name>/)")
    p.add_argument("--text-config", default="configs/data/text_tok.yaml")
    p.add_argument("--filter-config", default=None, help="en_kd quality-filter config yaml")
    p.add_argument("--jsonl", default="data/raw/text_anchor.jsonl", help="text_anchor input")
    p.add_argument("--min-sec", type=float, default=6.0, help="minimum en_solo crop length")
    p.add_argument("--seed-source", default="libriheavy", choices=["libriheavy", "local"],
                   help="seed_prompts clip source (local: --root = audio folder)")
    p.add_argument("--seed-sec", type=float, default=10.0, help="seed clip length in seconds")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="limit sample count for smoke tests")
    p.add_argument("--stage", choices=["build", "filter", "ingest"], default="build",
                   help="ingest: en_kd only — convert Colab ab_selfplay outputs (--root = dialogues dir)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list or not args.name:
        for name, cls in sorted(REGISTRY.items()):
            print(f"{name:18s} {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")
        return

    if args.name not in REGISTRY:
        raise SystemExit(f"Unknown dataset '{args.name}'. Registered: {sorted(REGISTRY)}")
    ds = REGISTRY[args.name].from_cli(args)

    if args.stage == "filter":
        if not hasattr(ds, "filter"):
            raise SystemExit(f"'{args.name}' has no filter stage (en_kd only)")
        ds.filter()
    elif args.stage == "ingest":
        if not hasattr(ds, "ingest_ab_selfplay"):
            raise SystemExit(f"'{args.name}' has no ingest stage (en_kd only)")
        ds.ingest_ab_selfplay(args.root)
    else:
        ds.build(limit=args.limit)


if __name__ == "__main__":
    main()
