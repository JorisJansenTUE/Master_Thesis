from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# ======================= EDIT THESE =======================
RESULTS_ROOT = Path("data/accessibility/base_mapping")
SOURCE = "facilities"   # "statent" or "facilities"
SCENARIOS = {        # First entry must be the baseline
    "Baseline": "baseline_00000",
    "Cluster": "cluster_00051",
    "Corridor": "corridor_00792",
}
MINUTES = 20
OUTFILE = RESULTS_ROOT / "sufficiency_20min_park5_comparison.png"
# ==========================================================

MODES = ("car", "bike", "pt")
MODE_LABELS = {"car": "Car", "bike": "Bicycle", "pt": "Public transport"}
COLORS = ("#2E6699", "#D46E33")  # two interventions
SHAPES = ("o", "D")


def read_summary(scenario_id: str) -> pd.DataFrame:
    suffix = "_facilities" if SOURCE == "facilities" else ""
    path = RESULTS_ROOT / f"results_{scenario_id}{suffix}" / "sufficiency_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run inspect_accessibility first.")
    return pd.read_csv(path)


def load_data() -> pd.DataFrame:
    if len(SCENARIOS) != 3:
        raise ValueError("SCENARIOS must contain baseline followed by exactly two interventions.")
    if SOURCE not in ("statent", "facilities"):
        raise ValueError("SOURCE must be 'statent' or 'facilities'.")

    records = []
    reference = None
    for label, scenario_id in SCENARIOS.items():
        summary = read_summary(scenario_id)
        summary["mode"] = summary["mode"].astype(str).str.lower()
        for mode in MODES:
            selection = summary.loc[(summary["mode"] == mode)
                                    & np.isclose(summary["time_min"], MINUTES)]
            if len(selection) != 1:
                raise ValueError(f"Expected exactly one {mode}/{MINUTES} min row "
                                 f"in {scenario_id}; found {len(selection)}")
            row = selection.iloc[0]
            thresholds = np.array([row["threshold_low_jobs"], row["threshold_high_jobs"]],
                                  dtype=float)
            if reference is None:
                reference = thresholds
            elif not np.allclose(reference, thresholds, rtol=1e-8, atol=1e-6):
                raise ValueError("The sufficiency thresholds differ: use a fixed baseline reference.")
            for level, field in (("Low", "pct_sufficient_low"),
                                 ("High", "pct_sufficient_high")):
                value = float(row[field])
                if not np.isfinite(value) or not (0 <= value <= 100):
                    raise ValueError(f"Invalid {field} for {scenario_id}/{mode}: {value}")
                records.append({"scenario": label, "scenario_id": scenario_id,
                                "mode": mode, "level": level, "pct": value,
                                "n_attached": int(row["n_attached"]),
                                "n_origins": int(row["n_origins"])})
    return pd.DataFrame(records)


def main() -> None:
    data = load_data()
    baseline_label, *scenario_labels = list(SCENARIOS)
    baseline = data[data["scenario"].eq(baseline_label)].set_index(["level", "mode"])["pct"]
    data["baseline_pct"] = [baseline.loc[(row.level, row.mode)]
                            for row in data.itertuples(index=False)]
    data["delta_pp"] = data["pct"] - data["baseline_pct"]

    subset = data[data["scenario"].ne(baseline_label)]
    max_abs_delta = float(subset["delta_pp"].abs().max())
    # Matching horizontal axes across panels make changes directly comparable.
    axis_limit = max(7.0, np.ceil((max_abs_delta + 4.0) / 2.0) * 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=True)
    y_positions = {"car": 2, "bike": 1, "pt": 0}
    level_settings = (("Low", "LOW SUFFICIENCY  ·  50% of baseline car mean"),
                      ("High", "HIGH SUFFICIENCY  ·  100% of baseline car mean"))
    offsets = (0.15, -0.15)

    for ax, (level, heading) in zip(axes, level_settings):
        ax.set_title(heading, fontsize=11, fontweight="bold", pad=17)
        ax.axvline(0, color="#374151", lw=1.5, zorder=1)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-0.60, 2.60)
        ax.set_yticks([y_positions[m] for m in MODES])
        ax.set_yticklabels([
            f"{MODE_LABELS[m]}\nBaseline: {baseline.loc[(level,m)]:g}%" for m in MODES
        ], fontsize=10)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.grid(axis="x", color="#e3e7ed", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}" if x else "0"))
        ax.set_xlabel("Change from baseline (percentage points)", fontsize=10, labelpad=10)

        for mode in MODES:
            y = y_positions[mode]
            ax.axhline(y + 0.48, color="#edf0f3", lw=0.9, zorder=0)
            for idx, scenario_label in enumerate(scenario_labels):
                row = data[(data["level"] == level)
                           & (data["mode"] == mode)
                           & (data["scenario"] == scenario_label)].iloc[0]
                delta = row["delta_pp"]
                yy = y + offsets[idx]
                ax.plot([0, delta], [yy, yy], lw=2.5, color=COLORS[idx],
                        alpha=0.75, solid_capstyle="round", zorder=2)
                ax.scatter([delta], [yy], s=86, marker=SHAPES[idx],
                           c=[COLORS[idx]], edgecolors="white", linewidths=0.8, zorder=3)
                # Label directly, so the size and direction of every difference are explicit.
                direction = 1 if delta >= 0 else -1
                label = f"{delta:+.1f} pp" if abs(delta-round(delta))>1e-8 else f"{delta:+.0f} pp"
                ax.annotate(label, (delta, yy), xytext=(direction*8, 0),
                            textcoords="offset points", ha="left" if direction > 0 else "right",
                            va="center", color=COLORS[idx], fontsize=9, fontweight="semibold")

    title = ("Job accessibility" if SOURCE == "statent" else "Work-facility accessibility")
    fig.suptitle(f"{MINUTES:g}-minute {title.lower()} — sufficiency change",
                 x=0.5, y=0.99, fontsize=16, weight="bold")
    legend = [Line2D([0], [0], color=COLORS[i], marker=SHAPES[i], markersize=8,
                     lw=2.5, label=scenario_labels[i]) for i in range(2)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               frameon=False, ncol=2, fontsize=11)

    n_attached = data[["scenario", "mode", "n_attached", "n_origins"]].drop_duplicates()
    attached_unchanged = n_attached.groupby("mode")["n_attached"].nunique().eq(1).all()
    denom_note = ("Attached-origin counts are unchanged across scenarios."
                  if attached_unchanged else
                  "Caution: attached-origin counts vary; percentages may use different origin sets.")
    fig.text(0.5, 0.035, "Sufficiency measured as share of attached origins.  " + denom_note,
             ha="center", fontsize=9, color="#5b6572")
    if os.getenv("DEMO_WATERMARK"):
        fig.text(0.5, 0.078, "ILLUSTRATIVE DEMO VALUES — NOT REAL SCENARIO RESULTS",
                 ha="center", fontsize=9, color="#a03a30", weight="bold")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.80, bottom=0.20, wspace=0.36)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=240, facecolor="white")
    fig.savefig(OUTFILE.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    table_path = OUTFILE.with_name(OUTFILE.stem + "_values.csv")
    data.to_csv(table_path, index=False)
    print(f"Saved: {OUTFILE}")
    print(f"Saved: {OUTFILE.with_suffix('.svg')}")
    print(f"Saved: {table_path}")
    print("\nChanges from baseline (percentage points):")
    print(subset.pivot_table(index=["mode", "level"], columns="scenario", values="delta_pp"))


if __name__ == "__main__":
    main()