import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model_patch import build_model_from_gates
from utils import build_dataloader, build_wikitext_splits, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IB-Transformer analysis plots.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--gate_checkpoint", type=str, required=True, help="Path to gate checkpoint.")
    parser.add_argument("--eval_summary", type=str, required=True, help="Path to evaluation_summary.json.")
    parser.add_argument("--prune_summary", type=str, required=True, help="Path to hard_prune_summary.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override.")
    return parser.parse_args()


def plot_rate_distortion(eval_summary, prune_summary, output_path: Path) -> None:
    baseline_ppl = eval_summary["wikitext_test"]["baseline"]["ppl"]
    ib_metrics = eval_summary["wikitext_test"]["ib_gate"]
    avg_gate = ib_metrics["avg_gate"]
    compression = 1.0 / max(avg_gate, 1e-8)
    hard_zero_ratio = prune_summary["hard_gate"]["zero_ratio"]
    hard_compression = 1.0 / max(1.0 - hard_zero_ratio, 1e-8)
    hard_ppl = prune_summary["hard_gate"]["ppl"]

    xs = [1.0, compression, hard_compression]
    ys = [baseline_ppl, ib_metrics["ppl"], hard_ppl]
    labels = ["Baseline", "IB-Gate (soft)", "IB-Gate (hard τ)"]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Compression rate (1 / average gate)")
    plt.ylabel("Perplexity")
    plt.title("Rate–Distortion Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "rate_distortion.png", dpi=200)
    plt.close()


def compute_gate_heatmap(model_bundle, config, device, output_path: Path) -> None:
    model = model_bundle.model.to(device)
    gates = model_bundle.gates
    tokenizer = model_bundle.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    num_layers = len(gates)
    seq_len = config["max_length"]
    gate_sums = torch.zeros(num_layers, seq_len, dtype=torch.double)
    gate_counts = torch.zeros(num_layers, seq_len, dtype=torch.double)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            model(**inputs)
            for layer_idx, gate in enumerate(gates):
                gate_tensor = gate.last_gate.detach().squeeze(-1)  # (batch, seq)
                gate_sums[layer_idx, :] += gate_tensor.sum(dim=0).cpu()
                gate_counts[layer_idx, :] += gate_tensor.size(0)

    heatmap = (gate_sums / gate_counts.clamp_min(1.0)).numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Average gate value")
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.yticks(range(num_layers), [f"L{idx+1}" for idx in range(num_layers)])
    plt.title("Gate Activation Heatmap (Validation set)")
    plt.tight_layout()
    plt.savefig(output_path / "gate_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_summary, "r", encoding="utf-8") as fp:
        eval_summary = json.load(fp)
    with open(args.prune_summary, "r", encoding="utf-8") as fp:
        prune_summary = json.load(fp)

    plot_rate_distortion(eval_summary, prune_summary, output_dir)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_model_from_gates(
        config["model_name"],
        args.gate_checkpoint,
        epsilon=config["hard_pruning"]["epsilon"],
    )
    compute_gate_heatmap(bundle, config, device, output_dir)


if __name__ == "__main__":
    main()
