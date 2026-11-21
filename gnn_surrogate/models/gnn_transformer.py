"""Spatial Graph Transformer -> Transformer Encoder surrogate.

Inspired by the best-performing architecture in Natterer et al. (2025):
  2 TransformerConv + GraphNorm layers, followed by 3 global Transformer
  Encoder layers, with 4 attention heads and 192-dimensional embeddings.

Predicts one-hour general-road link travel-time changes. Positional features (mid_x, mid_y) are
already included in the fixed ``x_static`` tensor.

The model returns signed travel-time changes (seconds). Add the fixed
baseline road travel times to reconstruct absolute travel times.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GraphNorm, TransformerConv
except ImportError as exc:
    raise ImportError(
        "gnn_transformer.py requires PyTorch Geometric. "
        "Install torch-geometric in your thesis environment."
    ) from exc


class GNNTransformer(nn.Module):
    """Predict signed, baseline-relative link travel times on a fixed dual graph.

    Args:
        static_dim: Number of selected static features (default 10).
        dynamic_dim: Number of scenario-specific features (default 9).
        baseline_dim: Number of baseline modal inputs (default 4).
        output_dim: Number of predicted road TT changes (default 1).
        hidden_dim: Node embedding width (Natterer: 192).
        heads: Attention heads for both stages (Natterer: 4).
        graph_layers: TransformerConv layer count (Natterer: 2).
        transformer_layers: Global Transformer Encoder layer count (Natterer: 3).
        feedforward_dim: Encoder MLP width (Natterer: 768).
        dropout: Dropout probability; configurable for Locarno.

    Forward arguments:
        x_static:    [N, static_dim]   shared across scenarios
        x_dynamic:   [N, dynamic_dim]  or [B, N, dynamic_dim]
        x_baseline:  [N, baseline_dim] log1p baseline counts for the target hour, shared
        edge_index:  [2, E]            fixed directed dual-graph edges

    Returns:
        [N, output_dim] or [B, N, output_dim], matching x_dynamic's rank.

    Batched scenarios remain independent in both graph message passing and
    global attention. Graph normalization uses a separate graph ID per sample.
    """

    def __init__(
        self,
        *,
        static_dim: int = 10,
        dynamic_dim: int = 9,
        baseline_dim: int = 4,
        output_dim: int = 1,
        hidden_dim: int = 192,
        heads: int = 4,
        graph_layers: int = 2,
        transformer_layers: int = 3,
        feedforward_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any(d <= 0 for d in (
            static_dim, dynamic_dim, baseline_dim, output_dim, hidden_dim,
            heads, graph_layers, transformer_layers, feedforward_dim,
        )):
            raise ValueError("All feature dimensions and layer counts must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.baseline_dim = baseline_dim
        self.output_dim = output_dim

        self.input_projection = nn.Linear(
            static_dim + dynamic_dim + baseline_dim, hidden_dim
        )

        # concat=True and per-head width hidden_dim/heads keeps each layer's
        # total output exactly hidden_dim (instead of heads*hidden_dim).
        self.graph_convs = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                beta=False,
                root_weight=True,
            )
            for _ in range(graph_layers)
        ])
        self.graph_norms = nn.ModuleList([
            GraphNorm(hidden_dim) for _ in range(graph_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x_static: Tensor,
        x_dynamic: Tensor,
        x_baseline: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        if x_static.ndim != 2 or x_static.shape[-1] != self.static_dim:
            raise ValueError(f"x_static must be [N, {self.static_dim}]")
        if x_baseline.ndim != 2 or x_baseline.shape[-1] != self.baseline_dim:
            raise ValueError(f"x_baseline must be [N, {self.baseline_dim}]")
        if x_dynamic.ndim not in (2, 3) or x_dynamic.shape[-1] != self.dynamic_dim:
            raise ValueError(f"x_dynamic must be [N, {self.dynamic_dim}] or [B,N,{self.dynamic_dim}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be int64 [2, E]")

        single = x_dynamic.ndim == 2
        if single:
            x_dynamic = x_dynamic.unsqueeze(0)

        b, n, _ = x_dynamic.shape
        if b < 1 or n < 1 or x_static.shape[0] != n or x_baseline.shape[0] != n:
            raise ValueError("All inputs must refer to the same nonempty set of N graph nodes")
        if not (x_static.device == x_dynamic.device == x_baseline.device == edge_index.device):
            raise ValueError("Move all model inputs to the same device")

        x = torch.cat((
            x_static.unsqueeze(0).expand(b, -1, -1),
            x_dynamic,
            x_baseline.unsqueeze(0).expand(b, -1, -1),
        ), dim=-1)
        h = F.relu(self.input_projection(x.reshape(b * n, -1)))

        # Disconnected copies of the same graph for PyG message passing.
        # No messages may travel between different intervention scenarios.
        if b == 1:
            batched_edges = edge_index
        else:
            offsets = (torch.arange(b, device=edge_index.device) * n).view(b, 1, 1)
            batched_edges = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        graph_ids = torch.arange(b, device=x.device).repeat_interleave(n)

        for conv, norm in zip(self.graph_convs, self.graph_norms):
            h = conv(h, batched_edges)
            h = F.relu(norm(h, graph_ids))
            h = self.dropout(h)

        # Full self-attention occurs within each scenario, never across batches.
        h = self.global_transformer(h.reshape(b, n, -1))
        predicted_delta = self.output_projection(h)
        return predicted_delta.squeeze(0) if single else predicted_delta
