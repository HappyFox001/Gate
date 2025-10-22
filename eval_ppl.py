import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_patch import build_model_from_gates
from utils import (
    build_dataloader,
    build_wikitext_splits,
    collate_lm_batch,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and IB-gated models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--gate_checkpoint", type=str, required=True, help="Path to saved gate state.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    return parser.parse_args()


def load_text_split(cfg: Dict) -> List[str]:
    dataset = load_dataset(
        cfg["name"],
        cfg.get("subset"),
        split=cfg["split"],
    )
    return dataset["text"]


def sentence_split(texts: List[str], delimiter: str) -> List[str]:
    sentences = []
    for text in texts:
        parts = text.split(delimiter)
        for idx, part in enumerate(parts[:-1]):
            fragment = part.strip()
            if fragment:
                sentences.append(fragment + delimiter)
        last = parts[-1].strip()
        if last:
            sentences.append(last)
    return sentences


def build_noise_sentences(
    original_sentences: List[str],
    noise_pool: List[str],
    interval: int,
    seed: int,
) -> List[str]:
    rng = torch.Generator().manual_seed(seed)
    noise_indices = torch.randint(
        low=0,
        high=len(noise_pool),
        size=(max(len(original_sentences) // interval, 1),),
        generator=rng,
    ).tolist()

    constructed = []
    noise_ptr = 0
    for idx in range(0, len(original_sentences), interval):
        chunk = original_sentences[idx : idx + interval]
        constructed.extend(chunk)
        if len(chunk) == interval and noise_pool:
            constructed.append(noise_pool[noise_indices[noise_ptr]])
            noise_ptr += 1
    return constructed


def texts_to_dataset(texts: List[str]) -> Dataset:
    return Dataset.from_dict({"text": texts})


def tokenize_texts(texts: List[str], tokenizer, block_size: int) -> Dataset:
    dataset = texts_to_dataset(texts)

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

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


def evaluate_language_model(
    model,
    dataloader: DataLoader,
    device,
    gates=None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    layer_gate_sums = torch.zeros(len(gates), dtype=torch.double) if gates else None
    layer_token_counts = torch.zeros(len(gates), dtype=torch.double) if gates else None

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            token_count = inputs["attention_mask"].sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            if gates:
                for idx, gate in enumerate(gates):
                    if gate.last_gate is None:
                        continue
                    gate_tensor = gate.last_gate.detach()
                    layer_gate_sums[idx] += gate_tensor.sum().item()
                    layer_token_counts[idx] += gate_tensor.numel()

    loss_per_token = total_loss / max(total_tokens, 1)
    ppl = math.exp(loss_per_token)

    if gates and layer_token_counts.sum() > 0:
        layer_means = layer_gate_sums / layer_token_counts.clamp_min(1.0)
        avg_gate = layer_means.mean().item()
    else:
        avg_gate = 1.0

    return {"loss": loss_per_token, "ppl": ppl, "avg_gate": avg_gate}


def select_pg19_subset(
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    num_documents: int,
) -> List[List[int]]:
    dataset = load_dataset("pg19", split="test")
    selected = []
    for entry in dataset:
        tokens = tokenizer(entry["text"], add_special_tokens=False)["input_ids"]
        if len(tokens) < min_tokens:
            continue
        clipped = tokens[:max_tokens]
        selected.append(clipped)
        if len(selected) == num_documents:
            break
    if len(selected) < num_documents:
        raise ValueError(f"Only found {len(selected)} PG-19 documents with at least {min_tokens} tokens.")
    return selected


def create_sliding_windows(tokens: List[int], window_size: int, stride: int) -> List[List[int]]:
    windows = []
    for start in range(0, len(tokens) - window_size + 1, stride):
        windows.append(tokens[start : start + window_size])
    if not windows:
        windows.append(tokens[-window_size:])
    return windows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["max_length"]

    block_size = config["max_length"]

    baseline_model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to(device)
    bundle = build_model_from_gates(config["model_name"], args.gate_checkpoint, epsilon=config["hard_pruning"]["epsilon"])
    ib_model = bundle.model.to(device)
    gates = bundle.gates

    datasets = build_wikitext_splits(
        config["train_dataset"],
        config["validation_dataset"],
        config["test_dataset"],
        tokenizer,
        block_size=block_size,
    )

    batch_size = config["training"]["batch_size"]
    test_loader = build_dataloader(datasets["test"], batch_size=batch_size, shuffle=False, drop_last=False)

    baseline_metrics = evaluate_language_model(baseline_model, test_loader, device)
    ib_metrics = evaluate_language_model(ib_model, test_loader, device, gates=gates)

    noise_config = config["noise_dataset"]
    train_texts = load_text_split(config["train_dataset"])
    test_texts = load_text_split(config["test_dataset"])
    noise_pool = sentence_split(train_texts, noise_config["sentence_delimiter"])
    original_sentences = sentence_split(test_texts, noise_config["sentence_delimiter"])
    noisy_sentences = build_noise_sentences(
        original_sentences,
        noise_pool,
        noise_config["insertion_interval"],
        noise_config["seed"],
    )
    noisy_dataset = tokenize_texts([" ".join(noisy_sentences)], tokenizer, block_size)
    noisy_loader = build_dataloader(noisy_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    baseline_noise = evaluate_language_model(baseline_model, noisy_loader, device)
    ib_noise = evaluate_language_model(ib_model, noisy_loader, device, gates=gates)

    pg_cfg = config["pg19_dataset"]
    pg_tokens = select_pg19_subset(
        tokenizer,
        min_tokens=pg_cfg["min_tokens"],
        max_tokens=pg_cfg["max_tokens"],
        num_documents=pg_cfg["num_documents"],
    )
    window_lengths = [1024, 2048, 3072, 4096]
    stride = 1024

    def dataset_from_windows(windows: List[List[int]]) -> Dataset:
        return Dataset.from_dict({"input_ids": windows})

    pg_results = defaultdict(dict)
    for window_size in window_lengths:
        window_sequences = []
        for doc_tokens in pg_tokens:
            window_sequences.extend(create_sliding_windows(doc_tokens, window_size, stride))
        pg_dataset = dataset_from_windows(window_sequences)
        pg_dataset.set_format(type="torch")
        loader = DataLoader(
            pg_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_lm_batch,
        )
        baseline_pg = evaluate_language_model(baseline_model, loader, device)
        ib_pg = evaluate_language_model(ib_model, loader, device, gates=gates)
        pg_results["baseline"][window_size] = baseline_pg["ppl"]
        pg_results["ib_gate"][window_size] = ib_pg["ppl"]

    eval_summary = {
        "wikitext_test": {
            "baseline": baseline_metrics,
            "ib_gate": ib_metrics,
        },
        "wikitext_noise": {
            "baseline": baseline_noise,
            "ib_gate": ib_noise,
            "noise_ppl_delta": {
                "baseline": baseline_noise["ppl"] - baseline_metrics["ppl"],
                "ib_gate": ib_noise["ppl"] - ib_metrics["ppl"],
            },
        },
        "pg19": pg_results,
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as fp:
        json.dump(eval_summary, fp, indent=2)


if __name__ == "__main__":
    main()
