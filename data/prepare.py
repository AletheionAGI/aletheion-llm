"""Utility script to build a simple text dataset from raw files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import datasets


def read_text_files(paths: Iterable[Path]) -> str:
    """Concatenate multiple text files into a single string."""

    contents = []
    for path in paths:
        contents.append(path.read_text(encoding="utf-8"))
    return "\n".join(contents)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [Path(p) for p in args.train]
    val_paths = [Path(p) for p in args.validation]
    test_paths = [Path(p) for p in args.test]

    dataset_dict = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"text": [read_text_files(train_paths)]}),
        "validation": datasets.Dataset.from_dict({"text": [read_text_files(val_paths)]}),
        "test": datasets.Dataset.from_dict({"text": [read_text_files(test_paths)]}),
    })

    dataset_dict.save_to_disk(str(output_dir))
    print(f"✅ Saved dataset to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset from raw text files")
    parser.add_argument("--train", nargs="+", required=True, help="Paths to training text files")
    parser.add_argument("--validation", nargs="+", required=True, help="Paths to validation text files")
    parser.add_argument("--test", nargs="+", required=True, help="Paths to test text files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the dataset")
    main(parser.parse_args())
