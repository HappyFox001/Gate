from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from ib_gates import GatedFeedForward, TokenGate


@dataclass(frozen=True)
class IBModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    gates: nn.ModuleList
    epsilon: float


def load_base_model(model_name: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False
    return model


def inject_ib_gates(
    model: PreTrainedModel,
    epsilon: float = 1e-3,
) -> nn.ModuleList:
    """
    Wraps each Transformer block FFN with a TokenGate as specified.
    Returns a ModuleList of gates ordered by layer index.
    """
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise ValueError("Expected model with a 'transformer' attribute (e.g., DistilGPT2).")
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        raise ValueError("Expected transformer blocks under attribute 'h'.")

    hidden_size = model.config.n_embd
    gates = nn.ModuleList()

    for layer_idx, block in enumerate(blocks):
        if not hasattr(block, "ffn"):
            raise ValueError(f"Transformer block at index {layer_idx} has no 'ffn' attribute to wrap.")
        gate = TokenGate(hidden_size=hidden_size, epsilon=epsilon)
        gates.append(gate)
        block.ffn = GatedFeedForward(block.ffn, gate)

    return gates


def build_ib_transformer(model_name: str, epsilon: float = 1e-3) -> IBModelBundle:
    model = load_base_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    gates = inject_ib_gates(model, epsilon=epsilon)
    return IBModelBundle(model=model, tokenizer=tokenizer, gates=gates, epsilon=epsilon)


def gate_parameters(gates: Iterable[TokenGate]) -> Iterable[nn.Parameter]:
    for gate in gates:
        yield from gate.parameters()


def save_gate_states(gates: Iterable[TokenGate]) -> list:
    return [gate.state_dict() for gate in gates]


def load_gate_states(gates: Iterable[TokenGate], states: Iterable[dict]) -> None:
    for gate, state in zip(gates, states):
        gate.load_state_dict(state)


def build_model_from_gates(model_name: str, gate_state_path: str, epsilon: float = 1e-3) -> IBModelBundle:
    bundle = build_ib_transformer(model_name, epsilon=epsilon)
    states = torch.load(gate_state_path, map_location="cpu")
    load_gate_states(bundle.gates, states)
    return bundle
