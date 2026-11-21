"""Lazy fixed-graph dataset with masked general-road travel-time changes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from gnn_surrogate.preprocessing.targets.eligibility import mask_from_inputs

DYNAMIC_FILE = "x_dynamic.npy"
DYNAMIC_METADATA = "dynamic_metadata.json"
TARGET_FILE = "target_delta.npz"
TARGET_METADATA = "target_metadata.json"
CHANNELS = ("road",)


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def discover_scenarios(dynamic_root: Path, target_root: Path) -> list[str]:
    dynamic_root, target_root = Path(dynamic_root), Path(target_root)
    if not dynamic_root.is_dir() or not target_root.is_dir():
        raise FileNotFoundError("Missing dynamic or target directory")
    ids = {p.name for p in dynamic_root.iterdir() if p.is_dir()} & {
        p.name for p in target_root.iterdir() if p.is_dir()}
    return sorted(sid for sid in ids if all((
        (dynamic_root / sid / DYNAMIC_FILE).is_file(),
        (dynamic_root / sid / DYNAMIC_METADATA).is_file(),
        (target_root / sid / TARGET_FILE).is_file(),
        (target_root / sid / TARGET_METADATA).is_file(),
    )))


def load_splits(path: Path) -> dict[str, list[str]]:
    manifest = _read_json(Path(path))
    splits = {}
    for name in ("train", "val", "test"):
        items = manifest.get(name)
        if not isinstance(items, list) or not items or len(items) != len(set(items)):
            raise ValueError(f"Invalid split: {name}")
        splits[name] = items
    all_ids = splits["train"] + splits["val"] + splits["test"]
    if len(all_ids) != len(set(all_ids)) or manifest.get("scenario_count") != len(all_ids):
        raise ValueError("Duplicate/mismatched scenario IDs across splits")
    return splits


class ScenarioDataset(Dataset):
    """Shared graph; each scenario loads [N,9] dynamics, [N,1] target + mask."""

    def __init__(self, graph_dir: Path, inputs_dir: Path,
                 dynamic_root: Path, target_root: Path,
                 scenario_ids: Sequence[str] | None = None) -> None:
        self.dynamic_root = Path(dynamic_root)
        self.target_root = Path(target_root)
        graph_dir, inputs_dir = Path(graph_dir), Path(inputs_dir)
        self.input_metadata = _read_json(inputs_dir / "input_metadata.json")
        with np.load(inputs_dir / "fixed_inputs.npz", allow_pickle=False) as file:
            static = file["x_static"]
            baseline_raw = file["baseline_hourly_raw"]
        with np.load(graph_dir / "baseline_tt.npz", allow_pickle=False) as file:
            baseline_tt = file["baseline_tt_s"]
            baseline_present = file["present_mask"]
            self.hour = int(file["hour"])
        edges = np.load(graph_dir / "dual_edge_index.npy", allow_pickle=False)
        n = static.shape[0]
        hours = self.input_metadata.get("hours", 30)
        if (static.ndim != 2 or baseline_raw.shape != (n, hours, 4)
                or not (0 <= self.hour < hours)
                or baseline_tt.shape != (n, 1)
                or baseline_present.shape != (n, 1)
                or self.input_metadata.get("travel_time_hour") != self.hour
                or self.input_metadata.get("node_count") != n
                or self.input_metadata.get("static_feature_names") is None
                or len(self.input_metadata["static_feature_names"]) != static.shape[1]):
            raise ValueError("Inconsistent fixed baseline inputs")
        if edges.ndim != 2 or edges.shape[0] != 2 or (edges.size and (edges.min() < 0 or edges.max() >= n)):
            raise ValueError("Invalid dual graph edges")
        if not all(np.isfinite(a).all() for a in (static, baseline_raw, baseline_tt)):
            raise ValueError("Nonfinite baseline features")
        # Use exactly the same hour as TRAVELTIME{hour}-{hour+1}avg.
        hourly = baseline_raw[:, self.hour, :]
        self.fixed = {
            "edge_index": torch.from_numpy(edges.astype(np.int64, copy=False)),
            "x_static": torch.from_numpy(static.astype(np.float32, copy=False)),
            "x_baseline": torch.from_numpy(np.log1p(hourly).astype(np.float32)),
            "baseline_tt_s": torch.from_numpy(baseline_tt.astype(np.float32)),
        }
        self.n_nodes = n
        self.target_channels = list(CHANNELS)
        if scenario_ids is None:
            self.scenario_ids = discover_scenarios(self.dynamic_root, self.target_root)
        else:
            self.scenario_ids = list(scenario_ids)
        if not self.scenario_ids or len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("Missing or duplicate scenarios")
        for sid in self.scenario_ids:
            for root, names in ((self.dynamic_root, (DYNAMIC_FILE, DYNAMIC_METADATA)),
                                (self.target_root, (TARGET_FILE, TARGET_METADATA))):
                if not all((root / sid / name).is_file() for name in names):
                    raise FileNotFoundError(f"Incomplete scenario {sid}: {root}")
        first = _read_json(self.dynamic_root / self.scenario_ids[0] / DYNAMIC_METADATA)
        self.dynamic_feature_names = first["feature_names"]
        self.static_feature_names = self.input_metadata["static_feature_names"]

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __getitem__(self, index: int) -> dict:
        sid = self.scenario_ids[index]
        d = _read_json(self.dynamic_root / sid / DYNAMIC_METADATA)
        t = _read_json(self.target_root / sid / TARGET_METADATA)
        if (d.get("scenario") != sid or d.get("node_count") != self.n_nodes
                or d.get("feature_names") != self.dynamic_feature_names
                or t.get("scenario") != sid
                or t.get("shape") != [self.n_nodes, 1]
                or t.get("channels") != self.target_channels
                or t.get("hour") != self.hour):
            raise ValueError(f"Scenario metadata mismatch: {sid}")
        x = np.load(self.dynamic_root / sid / DYNAMIC_FILE, allow_pickle=False)
        with np.load(self.target_root / sid / TARGET_FILE, allow_pickle=False) as file:
            y, mask = file["y_delta"], file["valid_mask"]
        if (x.shape != (self.n_nodes, len(self.dynamic_feature_names))
                or y.shape != (self.n_nodes, 1) or mask.shape != y.shape
                or mask.dtype != np.bool_
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError(f"Invalid tensors for {sid}")
        x_dynamic = torch.from_numpy(x.astype(np.float32, copy=False))
        input_mask = mask_from_inputs(
            self.fixed["x_static"], x_dynamic,
            self.static_feature_names, self.dynamic_feature_names,
        )
        if not torch.equal(input_mask, torch.from_numpy(mask)):
            mismatch = (input_mask != torch.from_numpy(mask)).sum().item()
            raise ValueError(
                f"{sid}: {mismatch} eligibility values derived from model inputs "
                "disagree with scenario network target masks"
            )
        return {"scenario_id": sid,
                "x_dynamic": x_dynamic,
                "y_delta": torch.from_numpy(y.astype(np.float32, copy=False)),
                "valid_mask": input_mask}
