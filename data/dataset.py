"""Dataset loading utilities for language modeling."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset):
    """Tokenized dataset for autoregressive language modeling."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        self.examples: List[List[int]] = []
        for text in texts:
            if not text:
                continue
            token_ids = tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
            )
            if len(token_ids) > 1:
                self.examples.append(token_ids)

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pad_token_id": torch.tensor(self.pad_token_id, dtype=torch.long),
        }


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length examples."""

    if not batch:
        raise ValueError("Batch must contain at least one example")

    pad_token_id = int(batch[0]["pad_token_id"]) if "pad_token_id" in batch[0] else 0
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding = max_len - seq_len

        input_pad = torch.full((padding,), pad_token_id, dtype=torch.long)
        label_pad = torch.full((padding,), -100, dtype=torch.long)

        input_ids.append(torch.cat([item["input_ids"], input_pad], dim=0))
        labels.append(torch.cat([item["labels"], label_pad], dim=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


def load_wikitext_dataset(
    tokenizer_name: str = "gpt2",
    dataset_config: str = "wikitext-2-raw-v1",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
):
    """Load WikiText splits and return tokenized datasets with a tokenizer."""

    from src.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(tokenizer_name, cache_dir=cache_dir)
    dataset = load_dataset("wikitext", dataset_config)

    train_dataset = TextDataset(tokenizer, dataset["train"]["text"], max_length)
    val_dataset = TextDataset(tokenizer, dataset["validation"]["text"], max_length)
    test_dataset = TextDataset(tokenizer, dataset["test"]["text"], max_length)

    return train_dataset, val_dataset, test_dataset, tokenizer


class TruthfulQADataset(Dataset):
    """Dataset for TruthfulQA question-answering evaluation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        questions: List[str],
        best_answers: List[List[str]],
        correct_answers: List[List[str]],
        incorrect_answers: List[List[str]],
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        self.questions = questions
        self.best_answers = best_answers
        self.correct_answers = correct_answers
        self.incorrect_answers = incorrect_answers

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]

        # Tokenize the question
        token_ids = self.tokenizer.encode(
            question,
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = torch.tensor(token_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "question": question,
            "best_answers": self.best_answers[idx],
            "correct_answers": self.correct_answers[idx],
            "incorrect_answers": self.incorrect_answers[idx],
            "pad_token_id": torch.tensor(self.pad_token_id, dtype=torch.long),
        }


def load_truthfulqa_dataset(
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
    split: str = "validation",
):
    """Load TruthfulQA dataset for evaluation.

    Args:
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        cache_dir: Cache directory for datasets
        split: Which split to load ('validation' is the main split)

    Returns:
        Tuple of (dataset, tokenizer)
    """
    from src.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(tokenizer_name, cache_dir=cache_dir)

    # Load TruthfulQA from HuggingFace
    dataset = load_dataset("truthful_qa", "generation", cache_dir=cache_dir)

    split_data = dataset[split]

    questions = split_data["question"]
    best_answers = split_data["best_answer"]
    correct_answers = split_data["correct_answers"]
    incorrect_answers = split_data["incorrect_answers"]

    # Convert best_answer (single string) to list format for consistency
    best_answers = [[ans] for ans in best_answers]

    truthfulqa_dataset = TruthfulQADataset(
        tokenizer=tokenizer,
        questions=questions,
        best_answers=best_answers,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers,
        max_length=max_length,
    )

    return truthfulqa_dataset, tokenizer
