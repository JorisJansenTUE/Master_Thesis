"""Quick LOCAL GNN-Transformer forward/backward check (not model training).

Run from project root:
    python -m gnn_surrogate.models.inspect_gnn_transformer

Uses a small, induced graph for CPU debugging. Do not treat its loss as a
training result or compare it with the full-network no-change benchmark.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.training.losses import masked_mse_loss
from .gnn_transformer import GNNTransformer
from preprocessing.utils import PROJECT_ROOT

# SETTINGS: 

GNN_DIR = PROJECT_ROOT / "data" / "GNN"
SPLITS_FILE = GNN_DIR / "splits_local.json"
SMOKE_NODES = 128       # Keep small for local CPU checks of full attention
SMOKE_BATCH_SIZE = 2    # Check scenarios do not mix in the Transformer
DEVICE = "cpu"         # Change to "cuda" only if a compatible GPU is available


def main() -> None:
    splits = load_splits(SPLITS_FILE)
    dataset = ScenarioDataset(
        GNN_DIR / "gnn_static",
        GNN_DIR / "gnn_inputs",
        GNN_DIR / "dynamic",
        GNN_DIR / "targets",
        scenario_ids=splits["train"],
    )
    fixed = dataset.fixed
    batch = next(iter(DataLoader(dataset, batch_size=min(SMOKE_BATCH_SIZE, len(dataset)))))
    n = min(SMOKE_NODES, dataset.n_nodes)

    # Subgraph with nodes 0..n-1 and only their mutual edges. Node order stays
    # unchanged, so no node-ID remapping is needed. This is a smoke test only.
    edges = fixed["edge_index"]
    edges = edges[:, (edges < n).all(dim=0)]
    if edges.shape[1] == 0:
        raise ValueError("No edges remain in this local subset; increase SMOKE_NODES")

    device = torch.device(DEVICE)
    model = GNNTransformer(
        static_dim=fixed["x_static"].shape[-1],
        dynamic_dim=batch["x_dynamic"].shape[-1],
        baseline_dim=fixed["x_baseline"].shape[-1],
        output_dim=batch["y_delta"].shape[-1],
    ).to(device)

    x_static = fixed["x_static"][:n].to(device)
    x_dynamic = batch["x_dynamic"][:, :n].to(device)
    x_baseline = fixed["x_baseline"][:n].to(device)
    y_delta = batch["y_delta"][:, :n].to(device)
    edges = edges.to(device)

    # Evaluation check: one scenario's prediction must not change when
    # another scenario is present in the same batch.
    model.eval()
    with torch.no_grad():
        batched_eval = model(x_static, x_dynamic, x_baseline, edges)
        single_eval = model(x_static, x_dynamic[0], x_baseline, edges)
        torch.testing.assert_close(batched_eval[0], single_eval, rtol=1e-4, atol=1e-5)

    model.train()
    pred = model(x_static, x_dynamic, x_baseline, edges)
    assert pred.shape == y_delta.shape
    assert torch.isfinite(pred).all()

    mask = batch["valid_mask"][:, :n].to(device)
    loss = masked_mse_loss(pred, y_delta, mask)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Local graph: {n:,} nodes, {edges.shape[1]:,} edges")
    print(f"Batch: {tuple(x_dynamic.shape)}, output: {tuple(pred.shape)}")
    print(f"Finite MSE loss: {loss.item():.6g}; gradients PASS")
    print("PASS: model/dataset interface, batch separation, shapes and backpropagation")
    print("NOTE: small-subgraph smoke test; not a full-network performance benchmark")


if __name__ == "__main__":
    main()
