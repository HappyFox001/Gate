import argparse
import math
from pathlib import Path

import torch

from model_patch import build_ib_transformer, gate_parameters
from utils import build_dataloader, build_wikitext_splits, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for IB-Gate pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g., cuda).")
    parser.add_argument("--batches", type=int, default=2, help="Number of validation batches to run.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_ib_transformer(config["model_name"], epsilon=config["hard_pruning"]["epsilon"])
    model = bundle.model.to(device)
    tokenizer = bundle.tokenizer
    gates = bundle.gates

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["max_length"]

    datasets = build_wikitext_splits(
        config["train_dataset"],
        config["validation_dataset"],
        config["test_dataset"],
        tokenizer,
        block_size=config["max_length"],
    )

    val_loader = build_dataloader(
        datasets["validation"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    print(f"Loaded DistilGPT-2 with {len(gates)} gates; total gate params: {sum(p.numel() for p in gate_parameters(gates))}")

    model.eval()
    with torch.no_grad():
        losses = []
        gate_means = []
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            losses.append(outputs.loss.item())
            per_layer = []
            for gate in gates:
                if gate.last_gate is not None:
                    per_layer.append(gate.last_gate.mean().item())
            gate_means.append(per_layer)

    avg_loss = sum(losses) / len(losses) if losses else math.nan
    print(f"Average validation loss over {len(losses)} batch(es): {avg_loss:.4f}")
    for idx, layer_means in enumerate(gate_means):
        formatted = ", ".join(f"{val:.3f}" for val in layer_means)
        print(f"Batch {idx} gate means: [{formatted}]")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
