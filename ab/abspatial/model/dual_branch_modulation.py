import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchCompetitiveModulation(nn.Module):
    def __init__(self, dim: int, alpha: float = 1.0, reduction: int = 4, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 可学习参数
        self.eps = eps

        hidden = max(dim // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.low_interact = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        self.low_modulation = nn.Linear(hidden, dim)
        self.low_attention = nn.Linear(hidden, dim)

        self.high_interact = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        self.high_modulation = nn.Linear(hidden, dim)
        self.high_attention = nn.Linear(hidden, dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.low_modulation.weight)
        nn.init.zeros_(self.low_modulation.bias)
        nn.init.zeros_(self.high_modulation.weight)
        nn.init.zeros_(self.high_modulation.bias)

        nn.init.zeros_(self.low_attention.weight)
        nn.init.zeros_(self.low_attention.bias)
        nn.init.zeros_(self.high_attention.weight)
        nn.init.zeros_(self.high_attention.bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(x).flatten(1)

    def _branch_forward(self, x_vec, y_vec, interact, modulation_head, attention_head):
        z = interact(torch.cat([x_vec, y_vec], dim=1))
        delta = self.alpha * torch.tanh(modulation_head(z))   # [-alpha, alpha]
        weight = torch.sigmoid(attention_head(z))             # [0, 1]
        return delta, weight

    def forward(self, x: torch.Tensor, y_low: torch.Tensor, y_high: torch.Tensor) -> torch.Tensor:
        x_vec = self._pool(x)
        y_low_vec = self._pool(y_low)
        y_high_vec = self._pool(y_high)

        delta_low, a_low = self._branch_forward(
            x_vec, y_low_vec,
            self.low_interact, self.low_modulation, self.low_attention
        )
        delta_high, a_high = self._branch_forward(
            x_vec, y_high_vec,
            self.high_interact, self.high_modulation, self.high_attention
        )

        a_sum = a_low + a_high + self.eps
        a_low_norm = a_low / a_sum
        a_high_norm = a_high / a_sum
        delta_gamma = a_low_norm * delta_low + a_high_norm * delta_high

        gamma = 1.0 + delta_gamma
        x_modulated = x * gamma.unsqueeze(-1).unsqueeze(-1)

        return x_modulated