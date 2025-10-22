import os
import random
from typing import Dict, List

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": []}
        result = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {"input_ids": result}

    chunked = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        load_from_cache_file=False,
    )
    chunked = chunked.filter(lambda example: len(example["input_ids"]) == block_size)
    return chunked


def build_wikitext_splits(
    train_cfg: Dict,
    val_cfg: Dict,
    test_cfg: Dict,
    tokenizer,
    block_size: int,
) -> DatasetDict:

    def load_split(split_cfg: Dict) -> Dataset:
        dataset = load_dataset(
            split_cfg["name"],
            split_cfg.get("subset"),
            split=split_cfg["split"],
        )
        return _tokenize_dataset(dataset, tokenizer, block_size)

    train = load_split(train_cfg)
    val = load_split(val_cfg)
    test = load_split(test_cfg)

    return DatasetDict({"train": train, "validation": val, "test": test})


def collate_lm_batch(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    input_batch = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.ones_like(input_batch, dtype=torch.bool)
    return {
        "input_ids": input_batch,
        "attention_mask": attention_mask,
        "labels": input_batch.clone(),
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_lm_batch,
    )
