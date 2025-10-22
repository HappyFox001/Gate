import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGate(nn.Module):
    """
    Token-wise scalar gate that modulates feed-forward outputs while leaving the
    residual pathway untouched. Parameters are a single weight vector and bias
    per layer, matching the specification.
    """

    def __init__(self, hidden_size: int, epsilon: float = 1e-3):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("epsilon", torch.tensor(epsilon), persistent=False)
        self.last_gate = None  # set during forward
        self.hard_threshold = None

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size) produced by the FFN.

        Returns:
            gated_states: Tensor shaped like hidden_states with token-wise gating applied.
        """
        logits = F.linear(hidden_states, self.weight.unsqueeze(0), self.bias)
        gates = torch.sigmoid(logits)  # (batch, seq_len, 1)
        self.last_gate = gates
        if self.hard_threshold is not None:
            mask = (gates >= self.hard_threshold).to(hidden_states.dtype)
            gated_states = mask * hidden_states
        else:
            # Soft interpolation with a stop-gradient branch to keep gradients stable.
            residual_branch = self.epsilon * hidden_states.detach()
            gated_states = gates * hidden_states + (1.0 - gates) * residual_branch
        return gated_states

    def mean_gate(self) -> torch.Tensor:
        """
        Returns the mean gate activation for the most recent forward pass.
        If forward has not been called, returns a scalar zero tensor.
        """
        if self.last_gate is None:
            return torch.zeros((), device=self.weight.device)
        return self.last_gate.mean()

    def reset_cache(self) -> None:
        self.last_gate = None

    def enable_hard_threshold(self, threshold: float) -> None:
        self.hard_threshold = threshold

    def disable_hard_threshold(self) -> None:
        self.hard_threshold = None


class GatedFeedForward(nn.Module):
    """
    Wraps an existing feed-forward network with a token gate.
    """

    def __init__(self, ffn: nn.Module, gate: TokenGate):
        super().__init__()
        self.ffn = ffn
        self.gate = gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return self.gate(ffn_output)
