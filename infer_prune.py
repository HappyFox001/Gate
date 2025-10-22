import argparse
import json
import time
from pathlib import Path

import torch

from model_patch import build_model_from_gates
from utils import build_dataloader, build_wikitext_splits, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard threshold gating inference benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--gate_checkpoint", type=str, required=True, help="Path to saved gate state.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for benchmark outputs.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold (defaults to config).")
    return parser.parse_args()


def evaluate_with_threshold(model, gates, dataloader, device, threshold=None):
    for gate in gates:
        if threshold is None:
            gate.disable_hard_threshold()
        else:
            gate.enable_hard_threshold(threshold)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    zero_counts = torch.zeros(len(gates), dtype=torch.double)

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            token_count = inputs["attention_mask"].sum().item()
            total_tokens += token_count
            total_loss += loss.item() * token_count
            if threshold is not None:
                for idx, gate in enumerate(gates):
                    gate_tensor = gate.last_gate.detach()
                    zero_counts[idx] += (gate_tensor < threshold).sum().item()
    elapsed = time.time() - start_time

    loss_per_token = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(loss_per_token)).item()
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    if threshold is None:
        zero_ratio = 0.0
    else:
        total_gate_decisions = total_tokens * len(gates)
        zero_ratio = (zero_counts.sum().item() / total_gate_decisions) if total_gate_decisions else 0.0

    return {"ppl": ppl, "tokens_per_sec": throughput, "zero_ratio": zero_ratio, "elapsed": elapsed}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_model_from_gates(config["model_name"], args.gate_checkpoint, epsilon=config["hard_pruning"]["epsilon"])
    model = bundle.model.to(device)
    gates = bundle.gates

    if bundle.tokenizer.pad_token is None:
        bundle.tokenizer.pad_token = bundle.tokenizer.eos_token

    datasets = build_wikitext_splits(
        config["train_dataset"],
        config["validation_dataset"],
        config["test_dataset"],
        bundle.tokenizer,
        block_size=config["max_length"],
    )

    dataloader = build_dataloader(
        datasets["test"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    soft_metrics = evaluate_with_threshold(model, gates, dataloader, device, threshold=None)
    threshold = args.threshold if args.threshold is not None else config["hard_pruning"]["threshold"]
    hard_metrics = evaluate_with_threshold(model, gates, dataloader, device, threshold=threshold)

    relative_throughput = (
        hard_metrics["tokens_per_sec"] / soft_metrics["tokens_per_sec"] if soft_metrics["tokens_per_sec"] > 0 else 0.0
    )

    summary = {
        "soft_gate": soft_metrics,
        "hard_gate": hard_metrics,
        "relative_throughput": relative_throughput,
        "threshold": threshold,
    }

    with open(output_dir / "hard_prune_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
