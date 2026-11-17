"""Lazy scenario dataset on one shared, fixed directed-street graph.

The graph and baseline tensors are loaded once per Dataset instance and live
in ``dataset.fixed``. ``dataset[i]`` loads ONLY that scenario's dynamic features
and baseline-relative target. This avoids duplicating large fixed tensors when
PyTorch batches multiple scenarios.

No MATSim XML, link mapping, intervention rules, or model architecture here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DYNAMIC_FILE = "x_dynamic.npy"
DYNAMIC_METADATA = "dynamic_metadata.json"
TARGET_FILE = "target_delta.npz"
TARGET_METADATA = "target_metadata.json"


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def discover_scenarios(dynamic_root: Path, target_root: Path) -> list[str]:
    """List scenarios with both feature and target artifacts present."""
    dynamic_root, target_root = Path(dynamic_root), Path(target_root)
    if not dynamic_root.is_dir() or not target_root.is_dir():
        raise FileNotFoundError("Both dynamic and target root directories must exist")
    names = {p.name for p in dynamic_root.iterdir() if p.is_dir()} & {
        p.name for p in target_root.iterdir() if p.is_dir()
    }
    return sorted(name for name in names if all((
        (dynamic_root / name / DYNAMIC_FILE).is_file(),
        (dynamic_root / name / DYNAMIC_METADATA).is_file(),
        (target_root / name / TARGET_FILE).is_file(),
        (target_root / name / TARGET_METADATA).is_file(),
    )))


def load_splits(path: Path) -> dict[str, list[str]]:
    """Read a reproducible train/val/test manifest and check for overlap."""
    manifest = _read_json(Path(path))
    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        items = manifest.get(name)
        if (not isinstance(items, list)
                or not all(isinstance(x, str) and x for x in items)
                or len(items) != len(set(items))):
            raise ValueError(f"Invalid {name} scenario list in {path}")
        splits[name] = items
    all_ids = splits["train"] + splits["val"] + splits["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A scenario appears in multiple data splits")
    if manifest.get("scenario_count") != len(all_ids):
        raise ValueError("Split manifest scenario_count does not match listed scenarios")
    return splits


class ScenarioDataset(Dataset):
    """Per-scenario tensors plus one shared fixed graph/baseline reference.

    Args:
        graph_dir: directory containing dual_edge_index.npy.
        inputs_dir: directory containing fixed_inputs.npz and input_metadata.json.
        dynamic_root: root of per-scenario dynamic/<scenario_id>/ files.
        target_root: root of per-scenario targets/<scenario_id>/ files.
        scenario_ids: optional ordered subset, e.g. load_splits(path)["train"].

    Args (temporal selection):
        temporal_mode: 'daily' (default), 'peak_hour', or 'hourly'.
        hour: required for 'peak_hour'; zero-indexed simulation hour.

    ``dataset.fixed`` contains four torch tensors, on CPU:
        edge_index             [2, E], int64
        x_static               [N, F_static], float32
        x_baseline             [N, M_input] or [N, T, M_input], log1p counts
        baseline_raw           [N, M_target] or [N, T, M_target], raw counts

    ``dataset[i]`` returns:
        scenario_id            string
        x_dynamic              [N, F_dynamic], float32
        y_delta                [N, M_target] or [N, T, M_target], signed changes

    All four baseline modal flows remain available as INPUTS. By default,
    daily targets exclude buses, whose daily link traversal counts are fixed
    by unchanged routes and schedules. The excluded bus changes are checked
    to be zero, so no unexpected scenario effects are silently ignored.
    Peak-hour/hourly targets include buses by default.

    Preprocessed hourly files stay unchanged. 'daily' sums ALL simulated
    hours (usually 30), not just the 24 calendar-day hours.
    """

    def __init__(
        self,
        graph_dir: Path,
        inputs_dir: Path,
        dynamic_root: Path,
        target_root: Path,
        scenario_ids: Sequence[str] | None = None,
        *,
        temporal_mode: str = "daily",
        hour: int | None = None,
        target_modes: Sequence[str] | None = None,
    ) -> None:
        if temporal_mode not in ("daily", "peak_hour", "hourly"):
            raise ValueError("temporal_mode must be 'daily', 'peak_hour', or 'hourly'")
        if temporal_mode == "peak_hour":
            if not isinstance(hour, int) or isinstance(hour, bool):
                raise ValueError("peak_hour requires an integer hour index")
        elif hour is not None:
            raise ValueError("hour is only used when temporal_mode='peak_hour'")
        self.temporal_mode = temporal_mode
        self.selected_hour = hour
        self.dynamic_root = Path(dynamic_root)
        self.target_root = Path(target_root)
        graph_dir, inputs_dir = Path(graph_dir), Path(inputs_dir)
        self.input_metadata = _read_json(inputs_dir / "input_metadata.json")

        with np.load(inputs_dir / "fixed_inputs.npz", allow_pickle=False) as saved:
            required = {"x_static", "x_baseline_hourly", "baseline_hourly_raw"}
            if not required.issubset(saved.files):
                raise ValueError(f"Missing fixed input tensors: {sorted(required - set(saved.files))}")
            static = np.asarray(saved["x_static"])
            baseline_input = np.asarray(saved["x_baseline_hourly"])
            baseline_raw = np.asarray(saved["baseline_hourly_raw"])

        edges = np.load(graph_dir / "dual_edge_index.npy", allow_pickle=False)
        if static.ndim != 2 or baseline_raw.ndim != 3:
            raise ValueError("Expected x_static [N,F] and baseline_hourly_raw [N,T,M]")
        n, f = static.shape
        t, m = baseline_raw.shape[1:]
        if (n == 0 or f == 0 or t == 0 or m == 0
                or baseline_raw.shape[0] != n or baseline_input.shape != baseline_raw.shape):
            raise ValueError("Inconsistent shapes in fixed baseline inputs")
        if (edges.ndim != 2 or edges.shape[0] != 2
                or not np.issubdtype(edges.dtype, np.integer)
                or (edges.size and (edges.min() < 0 or edges.max() >= n))):
            raise ValueError("Invalid dual_edge_index; expected [2,E] with node IDs in range")
        if (not all(np.issubdtype(x.dtype, np.number) and np.isfinite(x).all()
                    for x in (static, baseline_input, baseline_raw))
                or (baseline_raw < 0).any()):
            raise ValueError("Fixed baseline inputs must be finite, numeric, and nonnegative raw counts")
        if not np.allclose(baseline_input, np.log1p(baseline_raw), atol=1e-6):
            raise ValueError("Baseline transformed input does not equal log1p(raw flows)")
        if (self.input_metadata.get("node_count") != n
                or self.input_metadata.get("hours") != t
                or len(self.input_metadata.get("flow_modes", [])) != m
                or len(self.input_metadata.get("static_feature_names", [])) != f):
            raise ValueError("input_metadata.json is inconsistent with fixed tensors")

        if temporal_mode == "peak_hour" and not 0 <= hour < t:
            raise ValueError(f"hour must be between 0 and {t - 1}")

        self.input_modes = list(self.input_metadata["flow_modes"])
        if target_modes is None:
            target_modes = ([mode for mode in self.input_modes if mode != "buses"]
                            if temporal_mode == "daily" else self.input_modes)
        self.target_modes = list(target_modes)
        if (not self.target_modes
                or len(set(self.target_modes)) != len(self.target_modes)
                or any(mode not in self.input_modes for mode in self.target_modes)):
            raise ValueError("target_modes must be a nonempty, unique subset of baseline flow modes")
        self.target_mode_indices = [self.input_modes.index(mode) for mode in self.target_modes]
        self.check_fixed_daily_bus = (temporal_mode == "daily"
                                      and "buses" in self.input_modes
                                      and "buses" not in self.target_modes)

        # Aggregate raw counts FIRST, then apply log1p. Never sum log1p values.
        if temporal_mode == "daily":
            selected_baseline = baseline_raw.sum(axis=1, dtype=np.float64).astype(np.float32)
        elif temporal_mode == "peak_hour":
            selected_baseline = baseline_raw[:, hour, :]
        else:
            selected_baseline = baseline_raw

        self.fixed = {
            "edge_index": torch.from_numpy(edges.astype(np.int64, copy=False)),
            "x_static": torch.from_numpy(static.astype(np.float32, copy=False)),
            "x_baseline": torch.from_numpy(np.log1p(selected_baseline).astype(np.float32)),
            "baseline_raw": torch.from_numpy(
                np.ascontiguousarray(selected_baseline[..., self.target_mode_indices], dtype=np.float32)
            ),
        }
        self.n_nodes, self.hours, self.n_modes = n, t, m

        if scenario_ids is None:
            self.scenario_ids = discover_scenarios(self.dynamic_root, self.target_root)
        else:
            self.scenario_ids = list(scenario_ids)
            if (len(self.scenario_ids) != len(set(self.scenario_ids))
                    or any(not isinstance(x, str) or not x or Path(x).name != x
                           for x in self.scenario_ids)):
                raise ValueError("scenario_ids must be unique, nonempty directory names")
        if not self.scenario_ids:
            raise ValueError("No complete, matching scenario feature/target pairs found")
        for scenario_id in self.scenario_ids:
            if not all((
                (self.dynamic_root / scenario_id / DYNAMIC_FILE).is_file(),
                (self.dynamic_root / scenario_id / DYNAMIC_METADATA).is_file(),
                (self.target_root / scenario_id / TARGET_FILE).is_file(),
                (self.target_root / scenario_id / TARGET_METADATA).is_file(),
            )):
                raise FileNotFoundError(f"Incomplete preprocessing outputs for {scenario_id}")

        # Read one metadata file to establish feature order; each item validates it.
        first = _read_json(self.dynamic_root / self.scenario_ids[0] / DYNAMIC_METADATA)
        self.dynamic_feature_names = first.get("feature_names")
        if (not isinstance(self.dynamic_feature_names, list)
                or not self.dynamic_feature_names
                or len(self.dynamic_feature_names) != len(set(self.dynamic_feature_names))):
            raise ValueError("Missing or invalid dynamic feature names")

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __getitem__(self, index: int) -> dict:
        scenario_id = self.scenario_ids[index]
        dynamic_dir = self.dynamic_root / scenario_id
        target_dir = self.target_root / scenario_id

        dynamic_meta = _read_json(dynamic_dir / DYNAMIC_METADATA)
        target_meta = _read_json(target_dir / TARGET_METADATA)
        if (dynamic_meta.get("scenario") != scenario_id
                or dynamic_meta.get("node_count") != self.n_nodes
                or dynamic_meta.get("feature_names") != self.dynamic_feature_names):
            raise ValueError(f"Dynamic metadata mismatch: {scenario_id}")
        if (target_meta.get("scenario") != scenario_id
                or tuple(target_meta.get("shape", ())) != (self.n_nodes, self.hours, self.n_modes)
                or target_meta.get("flow_modes") != self.input_metadata["flow_modes"]):
            raise ValueError(f"Target metadata mismatch: {scenario_id}")

        x_dynamic = np.load(dynamic_dir / DYNAMIC_FILE, allow_pickle=False)
        with np.load(target_dir / TARGET_FILE, allow_pickle=False) as saved:
            if "y_delta" not in saved.files:
                raise ValueError(f"Missing y_delta in target file: {scenario_id}")
            y_delta = saved["y_delta"]
        if x_dynamic.shape != (self.n_nodes, len(self.dynamic_feature_names)):
            raise ValueError(f"Wrong dynamic feature shape: {scenario_id}")
        if y_delta.shape != (self.n_nodes, self.hours, self.n_modes):
            raise ValueError(f"Wrong target shape: {scenario_id}")
        if not (np.isfinite(x_dynamic).all() and np.isfinite(y_delta).all()):
            raise ValueError(f"Nonfinite scenario tensors: {scenario_id}")
        if self.temporal_mode == "daily":
            selected_delta = y_delta.sum(axis=1, dtype=np.float64).astype(np.float32)
        elif self.temporal_mode == "peak_hour":
            selected_delta = y_delta[:, self.selected_hour, :]
        else:
            selected_delta = y_delta

        if self.check_fixed_daily_bus:
            bus_idx = self.input_modes.index("buses")
            if np.any(np.abs(selected_delta[..., bus_idx]) > 1e-4):
                raise ValueError(
                    f"Daily bus flows changed in {scenario_id}; cannot exclude buses "
                    "from training targets without investigating the scenario."
                )

        selected_delta = np.ascontiguousarray(
            selected_delta[..., self.target_mode_indices], dtype=np.float32
        )
        if np.any(selected_delta + self.fixed["baseline_raw"].numpy() < -1e-5):
            raise ValueError(f"Negative reconstructed absolute traffic: {scenario_id}")

        return {
            "scenario_id": scenario_id,
            "x_dynamic": torch.from_numpy(x_dynamic.astype(np.float32, copy=False)),
            "y_delta": torch.from_numpy(selected_delta),
        }
