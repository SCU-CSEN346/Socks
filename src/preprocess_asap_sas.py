"""CLI for preparing ASAP-SAS into the canonical ASAG baseline schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from .asag_data import (
    DEFAULT_ASAP_SAS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_VAL_SIZE,
    build_asap_sas_splits,
    write_processed_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ASAP-SAS into canonical ASAG baseline splits."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ASAP_SAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, score_ranges = build_asap_sas_splits(
        data_dir=args.data_dir,
        val_size=args.val_size,
        seed=args.seed,
    )
    paths = write_processed_splits(splits, score_ranges, output_dir=args.output_dir)

    print("ASAP-SAS preprocessing complete.")
    for split_name, df in splits.items():
        labeled = int(df["score_raw"].notna().sum())
        print(f"- {split_name}: {len(df)} rows ({labeled} labeled)")
    print("\nOutputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
