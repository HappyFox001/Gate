import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from model_patch import build_ib_transformer, gate_parameters, save_gate_states
from utils import (
    build_dataloader,
    build_wikitext_splits,
    ensure_dir,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IB gates on a frozen DistilGPT-2 backbone.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g., cuda).",
    )
    return parser.parse_args()


def current_lambda_weight(
    step: int,
    total_steps: int,
    warmup_fraction: float,
    ramp_end_fraction: float,
    target: float,
) -> float:
    if total_steps == 0:
        return target
    progress = step / total_steps
    if progress < warmup_fraction:
        return 0.0
    if progress < ramp_end_fraction:
        span = max(ramp_end_fraction - warmup_fraction, 1e-8)
        return target * ((progress - warmup_fraction) / span)
    return target


def evaluate(
    model,
    gates,
    dataloader,
    device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    layer_gate_sums = torch.zeros(len(gates), dtype=torch.double)
    layer_token_counts = torch.zeros(len(gates), dtype=torch.double)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            token_count = inputs["attention_mask"].sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            for idx, gate in enumerate(gates):
                if gate.last_gate is None:
                    continue
                gate_tensor = gate.last_gate.detach()
                layer_gate_sums[idx] += gate_tensor.sum().item()
                layer_token_counts[idx] += gate_tensor.numel()

    model.train()

    loss_per_token = total_loss / max(total_tokens, 1)
    ppl = math.exp(loss_per_token)

    if layer_token_counts.sum() == 0:
        avg_gate = 0.0
    else:
        layer_means = layer_gate_sums / layer_token_counts.clamp_min(1.0)
        avg_gate = layer_means.mean().item()

    return {"loss": loss_per_token, "ppl": ppl, "avg_gate": avg_gate}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    log_path = Path(args.output_dir) / "train_log.jsonl"

    set_seed(config["seed"])

    epsilon = config["hard_pruning"]["epsilon"]
    bundle = build_ib_transformer(config["model_name"], epsilon=epsilon)
    model = bundle.model
    tokenizer = bundle.tokenizer
    gates = bundle.gates

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    existing_max = getattr(tokenizer, "model_max_length", config["max_length"])
    if existing_max is None or existing_max > 10**8:
        tokenizer.model_max_length = config["max_length"]
    else:
        tokenizer.model_max_length = max(config["max_length"], existing_max)

    datasets = build_wikitext_splits(
        config["train_dataset"],
        config["validation_dataset"],
        config["test_dataset"],
        tokenizer,
        block_size=config["max_length"],
    )

    train_loader = build_dataloader(
        datasets["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    val_loader = build_dataloader(
        datasets["validation"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gate_params = list(gate_parameters(gates))
    optimizer = AdamW(
        gate_params,
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / config["training"]["gradient_accumulation_steps"])
    total_update_steps = num_update_steps_per_epoch * config["training"]["epochs"]
    warmup_steps = int(total_update_steps * config["training"]["warmup_fraction"])

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    grad_accum = config["training"]["gradient_accumulation_steps"]
    lambda_target = config["training"]["lambda_ib"]
    lambda_warmup = config["training"]["lambda_warmup_fraction"]
    lambda_ramp_end = config["training"]["lambda_ramp_end_fraction"]
    mixed_precision = config["training"]["mixed_precision"].lower() == "fp16"
    grad_clip = config["training"]["grad_clip"]

    eval_interval_fraction = config["training"]["eval_interval_fraction"]
    eval_interval_steps = max(1, int(num_update_steps_per_epoch * eval_interval_fraction))

    optimizer.zero_grad()
    train_log = open(log_path, "w", encoding="utf-8")

    for epoch in range(config["training"]["epochs"]):
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=mixed_precision):
                outputs = model(**inputs)
                lm_loss = outputs.loss
                gate_means = [gate.last_gate.mean() for gate in gates if gate.last_gate is not None]
                if gate_means:
                    gate_loss = torch.stack(gate_means).mean()
                else:
                    gate_loss = torch.tensor(0.0, device=device)
                lambda_weight = current_lambda_weight(
                    global_step,
                    total_update_steps,
                    lambda_warmup,
                    lambda_ramp_end,
                    lambda_target,
                )
                loss = lm_loss + lambda_weight * gate_loss

            loss = loss / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(gate_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

                global_step += 1

                log_entry = {
                    "epoch": epoch,
                    "step": global_step,
                    "lm_loss": lm_loss.item(),
                    "gate_loss": gate_loss.item(),
                    "lambda": lambda_weight,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                train_log.write(json.dumps(log_entry) + "\n")
                train_log.flush()

                if global_step % eval_interval_steps == 0:
                    metrics = evaluate(model, gates, val_loader, device)
                    eval_entry = {
                        "epoch": epoch,
                        "step": global_step,
                        "val_ppl": metrics["ppl"],
                        "val_gate": metrics["avg_gate"],
                    }
                    train_log.write(json.dumps(eval_entry) + "\n")
                    train_log.flush()

    train_log.close()
    gate_state = save_gate_states(gates)
    torch.save(gate_state, Path(args.output_dir) / "ib_gates.pt")


if __name__ == "__main__":
    main()
