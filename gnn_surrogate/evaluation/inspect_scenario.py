"""Inspect reconstructed travel times from the congestion-delay surrogate.

Run from the project root:
    python -m gnn_surrogate.evaluation.inspect_scenario

Configure paths and SCENARIO_ID below; no command-line arguments required.
Works with the road-only congestion-delay-change surrogate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_surrogate.dataset.dataset import ScenarioDataset
from gnn_surrogate.preprocessing.targets.target_tensors import _eligible_links
from gnn_surrogate.evaluation.delay_reconstruction import reconstruct_road_travel_times


# ---------------------------- LOCAL SETTINGS -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_DIR = Path("/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_tt_delay")
SIMULATIONS_DIR = Path("/vast.mnt/home/20201733/locarno_ETH_Model/training_dataset_v4")
RUN_TAG = "tt_delay_001"
SCENARIO_ID = "corridor_00828"
BASELINE_ID = "baseline_00000"
CHECKPOINT = GNN_DIR / "runs" / RUN_TAG / "best.pt"
OUTPUT_DIR = GNN_DIR / "runs" / RUN_TAG / "scenario_inspection" / SCENARIO_ID
DEVICE = "cpu"   # Set to 'cuda' if a GPU is allocated.
TOP_N = 30
MAKE_PLOTS = True
# ---------------------------------------------------------------------------


def load_prediction(dataset: ScenarioDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from gnn_surrogate.models.gnn_transformer import GNNTransformer
    device = torch.device(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if ckpt.get("static_feature_names") != dataset.static_feature_names:
        raise ValueError("Checkpoint/static feature mismatch")
    if ckpt.get("target_channels") != dataset.target_channels:
        raise ValueError("Checkpoint/output channel mismatch")
    model = GNNTransformer(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    item = dataset[0]
    fixed = dataset.fixed
    with torch.inference_mode():
        pred = model(
            fixed["x_static"].to(device), item["x_dynamic"].to(device),
            fixed["x_baseline"].to(device), fixed["edge_index"].to(device),
        ).cpu().numpy()
    return (pred, item["y_delta"].numpy(), item["valid_mask"].numpy(),
            fixed["baseline_tt_s"].numpy())


def build_link_table(dataset: ScenarioDataset, pred: np.ndarray, truth: np.ndarray,
                     mask: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """Build link-level diagnostics with correct total-travel-time reconstruction."""
    n, c = truth.shape
    if c != 1 or pred.shape != (n, 1) or mask.shape != (n, 1) or baseline.shape != (n, 1):
        raise ValueError("Road-only prediction, target, mask, and baseline must be [N,1]")
    with np.load(GNN_DIR / "gnn_static" / "baseline_tt.npz", allow_pickle=False) as cache:
        baseline_ff = cache["baseline_freeflow_s"]
        if int(cache["hour"]) != dataset.hour:
            raise ValueError("Baseline free-flow hour mismatch")
    x_dynamic = dataset[0]["x_dynamic"].numpy()
    reconstruction = reconstruct_road_travel_times(
        baseline, baseline_ff, x_dynamic, dataset.dynamic_feature_names,
        pred, mask, actual_delay_delta_s=truth,
    )
    valid = mask[:, 0]
    rows = pd.DataFrame({
        "node_id": np.arange(n)[valid],
        "channel": 0,
        "channel_name": "road",
        "baseline_tt_s": baseline[valid, 0],
        "baseline_freeflow_tt_s": baseline_ff[valid, 0],
        "actual_delay_change_s": truth[valid, 0],
        "predicted_delay_change_s": pred[valid, 0],
        "freeflow_change_s": reconstruction["freeflow_change_s"][valid, 0],
        "scenario_freeflow_reconstructed_s": reconstruction["scenario_freeflow_s"][valid, 0],
        "actual_delta_s": reconstruction["actual_total_delta_s"][valid, 0],
        "predicted_delta_s": reconstruction["predicted_total_delta_s"][valid, 0],
        "actual_tt_s": reconstruction["actual_tt_s"][valid, 0],
        "predicted_tt_s": reconstruction["predicted_tt_s"][valid, 0],
    })
    rows["error_s"] = rows.predicted_tt_s - rows.actual_tt_s
    rows["abs_error_s"] = rows.error_s.abs()
    rows["squared_error_s2"] = rows.error_s ** 2

    nodes = pd.read_csv(GNN_DIR / "gnn_static" / "dual_nodes.csv.gz",
                        dtype={"segment_id": str, "from_node": str, "to_node": str})
    rows = rows.merge(nodes[["node_id", "segment_id", "from_node", "to_node",
                             "mid_x", "mid_y", "osm_way_id"]],
                      on="node_id", how="left", validate="one_to_one")

    scenario = SIMULATIONS_DIR / SCENARIO_ID
    eligible = _eligible_links(nodes, scenario, dataset.hour)
    eligible = eligible[["node_id", "link_id", "tt_s", "FREESPEED"]]
    rows = rows.merge(eligible, on="node_id", how="left", validate="one_to_one")
    if rows.link_id.isna().any():
        raise ValueError("Eligible target not found in MATSim link mapping")
    if (rows.actual_tt_s - rows.tt_s).abs().max() > 0.02:
        raise ValueError("Reconstructed MATSim scenario time differs from linkstats; "
                         "check free-flow-change inputs and mapping")

    hour = dataset.hour
    volume_col = f"HRS{hour}-{hour+1}avg"
    stats = pd.read_csv(scenario / "targets" / "linkstats.txt.gz", sep="\t",
                        usecols=["LINK", "LENGTH", "CAPACITY", volume_col], dtype={"LINK": str})
    rows = rows.merge(stats, left_on="link_id", right_on="LINK", how="left",
                      validate="one_to_one")
    rows["freeflow_tt_s"] = rows.LENGTH / rows.FREESPEED
    if (rows.scenario_freeflow_reconstructed_s - rows.freeflow_tt_s).abs().max() > 0.02:
        raise ValueError("Dynamic free-speed change does not reconstruct scenario "
                         "free-flow times. Evaluate from MATSim scenario geometry instead.")
    rows = rows.rename(columns={volume_col: "hourly_volume_avg",
                                "CAPACITY": "scenario_capacity"}).drop(columns=["LINK", "tt_s"])

    rows["baseline_ref_link_id"] = rows.link_id
    base_stats = pd.read_csv(SIMULATIONS_DIR / BASELINE_ID / "targets" / "linkstats.txt.gz",
                             sep="\t", dtype={"LINK": str},
                             usecols=["LINK", "LENGTH", "FREESPEED", "CAPACITY",
                                      volume_col, f"TRAVELTIME{hour}-{hour+1}avg"])
    base_stats = base_stats.rename(columns={
        "LINK": "baseline_ref_link_id", "LENGTH": "baseline_length_m",
        "FREESPEED": "baseline_freespeed_m_s", "CAPACITY": "baseline_capacity",
        volume_col: "baseline_volume_avg",
        f"TRAVELTIME{hour}-{hour+1}avg": "baseline_linkstats_tt_s"})
    rows = rows.merge(base_stats, on="baseline_ref_link_id", how="left",
                      validate="many_to_one")
    if rows.baseline_linkstats_tt_s.isna().any():
        raise ValueError("Baseline linkstats missing for eligible road links")
    if (rows.baseline_linkstats_tt_s - rows.baseline_tt_s).abs().max() > 0.02:
        raise ValueError("Baseline linkstats and cache do not agree")
    rows["volume_change_avg"] = rows.hourly_volume_avg - rows.baseline_volume_avg
    rows["actual_delay_s"] = rows.actual_tt_s - rows.freeflow_tt_s
    rows["predicted_delay_s"] = rows.predicted_tt_s - rows.freeflow_tt_s

    interventions = json.loads((scenario / "interventions.json").read_text())
    by_segment: dict[str, set[str]] = {}
    for intervention in interventions:
        by_segment.setdefault(str(intervention["segment_id"]), set()).add(
            str(intervention["intervention_type"])
        )
    rows["intervention"] = rows.segment_id.map(
        lambda seg: ",".join(sorted(by_segment.get(str(seg), set())))
    )
    rows["directly_modified"] = rows.intervention.ne("")
    return rows.sort_values("squared_error_s2", ascending=False)


def report(rows: pd.DataFrame, hour: int) -> dict:
    result = {"scenario": SCENARIO_ID, "hour": hour,
              "channels": {}, "output_dir": str(OUTPUT_DIR)}
    for channel, group in rows.groupby("channel_name", sort=False):
        err = group.error_s.to_numpy(dtype=float)
        actual = group.actual_delta_s.to_numpy(dtype=float)
        ss = np.sum(err ** 2)
        sst = np.sum((actual - actual.mean()) ** 2)
        metrics = {
            "eligible_links": int(len(group)),
            "mae_s": float(np.abs(err).mean()),
            "rmse_s": float(np.sqrt(np.mean(err ** 2))),
            "no_change_rmse_s": float(np.sqrt(np.mean(actual ** 2))),
            "unchanged_delay_rmse_s": float(np.sqrt(np.mean(group.actual_delay_change_s.to_numpy(dtype=float) ** 2))),
            "r2_delta": float(1 - ss / sst) if sst > 0 else None,
            "mean_error_s": float(err.mean()),
            "top_1_link_share_of_squared_error_pct": float(100 * group.squared_error_s2.iloc[0] / ss) if ss > 0 else 0.0,
            "top_10_links_share_of_squared_error_pct": float(100 * group.squared_error_s2.head(10).sum() / ss) if ss > 0 else 0.0,
            "actual_tt_over_120s": int((group.actual_tt_s > 120).sum()),
            "actual_delay_over_60s": int((group.actual_delay_s > 60).sum()),
            "predicted_below_scenario_freeflow": int((group.predicted_tt_s < group.freeflow_tt_s - 0.01).sum()),
        }
        result["channels"][channel] = metrics
        print(f"\n=== {SCENARIO_ID} / {channel} ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        columns = ["link_id", "node_id", "actual_delta_s", "predicted_delta_s",
                   "error_s", "baseline_tt_s", "actual_tt_s", "predicted_tt_s",
                   "freeflow_tt_s", "baseline_volume_avg", "hourly_volume_avg",
                   "freeflow_change_s", "intervention"]
        print("\nWorst links by squared prediction error:")
        print(group.head(min(TOP_N, len(group)))[columns].to_string(index=False,
              float_format=lambda x: f"{x:.2f}"))
    return result


def plots(rows: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for channel, group in rows.groupby("channel_name", sort=False):
        # Full-range scatter preserves any genuinely extreme congestion cases.
        fig, ax = plt.subplots(figsize=(7.0, 6.4))
        ax.scatter(group.actual_delta_s, group.predicted_delta_s,
                   s=7, alpha=0.22, linewidths=0, rasterized=True)
        lo = min(group.actual_delta_s.min(), group.predicted_delta_s.min(), 0)
        hi = max(group.actual_delta_s.max(), group.predicted_delta_s.max(), 0)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
        ax.set(xlabel="Actual Δ travel time (s)", ylabel="Predicted Δ travel time (s)",
               title=f"{SCENARIO_ID} — {channel}")
        ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"scatter_{channel}.png", dpi=160)
        plt.close(fig)

        top = group.head(min(20, len(group))).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        y = np.arange(len(top))
        ax.barh(y - 0.2, top.actual_delta_s, height=0.4, label="MATSim")
        ax.barh(y + 0.2, top.predicted_delta_s, height=0.4, label="GNN")
        ax.set_yticks(y, top.link_id.astype(str))
        ax.set(xlabel="Change in 08:00–09:00 link travel time (s)",
               title=f"20 worst links — {SCENARIO_ID} / {channel}")
        ax.grid(axis="x", alpha=0.2); ax.legend(); fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"worst_links_{channel}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    scenario = SIMULATIONS_DIR / SCENARIO_ID
    for required in (CHECKPOINT, scenario / "link_mapping.csv.gz",
                     scenario / "targets" / "linkstats.txt.gz",
                     SIMULATIONS_DIR / BASELINE_ID / "targets" / "linkstats.txt.gz"):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {required} (change settings at top)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_meta = json.loads((GNN_DIR / "targets" / SCENARIO_ID / "target_metadata.json").read_text())
    if target_meta.get("target_type") != "road_congestion_delay_change":
        raise ValueError("Inspector requires congestion-delay targets, not total-time changes")
    data = ScenarioDataset(GNN_DIR / "gnn_static", GNN_DIR / "gnn_inputs",
                           GNN_DIR / "dynamic", GNN_DIR / "targets", [SCENARIO_ID])
    pred, truth, mask, baseline = load_prediction(data)
    rows = build_link_table(data, pred, truth, mask, baseline)
    rows.to_csv(OUTPUT_DIR / "link_errors.csv.gz", index=False, compression="gzip")
    summary = report(rows, data.hour)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if MAKE_PLOTS:
        plots(rows)
    print(f"\nSaved link_errors.csv.gz, summary.json and plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
