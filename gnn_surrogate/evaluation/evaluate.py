"""Evaluate the general-road congestion-delay surrogate in travel-time seconds.

Targets contain congestion-delay CHANGES, but the main evaluation compares
reconstructed TOTAL travel-time changes with MATSim. All statistics and plots
exclude ineligible links. Two benchmarks are reported:
  - no_change: total travel time remains at baseline (delta TT = 0)
  - unchanged_delay: free-flow changes are known, delay stays at baseline
The latter is the natural zero-output baseline for the delay surrogate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.evaluation.delay_reconstruction import reconstruct_road_travel_times
from gnn_surrogate.models.gnn_transformer import GNNTransformer
from preprocessing.utils import PROJECT_ROOT

CHANNEL = "road"
MAGNITUDE_EDGES_S = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
MAGNITUDE_LABELS = ("0", "0–1", "1–5", "5–20", "20–100", ">100")
BASELINE_DELAY_EDGES_S = np.array([0.0, 5.0, 15.0, 60.0])
BASELINE_DELAY_LABELS = ("≤0", "0–5", "5–15", "15–60", ">60")
SCATTER_MAX_POINTS = 50_000
TARGET_TYPE = "road_congestion_delay_change"


def _empty_stats() -> dict:
    return {"n": 0, "abs": 0.0, "sq": 0.0, "sq_no_change": 0.0,
            "sq_unchanged_delay": 0.0, "sum_total": 0.0, "sum_total2": 0.0,
            "sum_delay": 0.0, "sum_delay2": 0.0, "sum_absolute": 0.0,
            "sum_absolute2": 0.0, "bias": 0.0, "below_ff": 0}


def _metrics(s: dict) -> dict:
    n = s["n"]
    if not n:
        return {"count": 0, "mae_s": None, "rmse_s": None, "r2_delta": None,
                "r2_delay_delta": None, "r2_absolute_tt": None,
                "no_change_rmse_s": None, "unchanged_delay_rmse_s": None,
                "mean_error_s": None, "predicted_below_freeflow": 0}

    def r2(sum_y: float, sum_y2: float):
        sst = sum_y2 - sum_y ** 2 / n
        return 1.0 - s["sq"] / sst if sst > 1e-12 else None

    return {
        "count": n,
        "mae_s": s["abs"] / n,
        "rmse_s": np.sqrt(s["sq"] / n).item(),
        "r2_delta": r2(s["sum_total"], s["sum_total2"]),
        "r2_delay_delta": r2(s["sum_delay"], s["sum_delay2"]),
        "r2_absolute_tt": r2(s["sum_absolute"], s["sum_absolute2"]),
        "no_change_rmse_s": np.sqrt(s["sq_no_change"] / n).item(),
        "unchanged_delay_rmse_s": np.sqrt(s["sq_unchanged_delay"] / n).item(),
        "mean_error_s": s["bias"] / n,
        "predicted_below_freeflow": s["below_ff"],
    }


def _plot_history(path: Path, out: Path, best_epoch: int, plt) -> None:
    if not path.is_file():
        print(f"No loss curve: {path} not found", flush=True)
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"epoch", "train_mse", "val_mse"}.issubset(rows[0]):
        print(f"No loss curve: {path} has no epoch/train_mse/val_mse columns", flush=True)
        return
    epoch = np.array([int(r["epoch"]) for r in rows])
    train = np.array([float(r["train_mse"]) for r in rows])
    val = np.array([float(r["val_mse"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5.1))
    ax.plot(epoch, train, lw=2, label="Training")
    ax.plot(epoch, val, lw=2, label="Validation")
    if best_epoch in epoch:
        ax.axvline(best_epoch, ls="--", color="0.4", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Masked MSE of Δ congestion delay (s²)",
           title="Training and validation loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_train_validation.png", dpi=180)
    plt.close(fig)


def _plot_scatter(scatter: list, metrics: dict, out: Path, plt) -> None:
    if not scatter:
        return
    actual = np.concatenate([a for a, _ in scatter])
    pred = np.concatenate([b for _, b in scatter])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(actual, pred, s=5, alpha=0.12, linewidths=0, rasterized=True)
    lo, hi = min(actual.min(), pred.min(), 0), max(actual.max(), pred.max(), 0)
    if hi <= lo:
        hi = lo + 1
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.2", label="Perfect prediction")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.set(xlabel="Actual total travel-time change (s)",
           ylabel="Predicted total travel-time change (s)",
           title="Road links — reconstructed total travel-time changes",
           xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal", adjustable="box")
    details = f"RMSE: {metrics['rmse_s']:.2f} s"
    if metrics["r2_delta"] is not None:
        details += f"\nR²: {metrics['r2_delta']:.3f}"
    ax.text(.03, .97, details, va="top", transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "scatter_road.png", dpi=180)
    plt.close(fig)


def _plot_bins(bin_data: dict, labels: tuple[str, ...], name: str,
               title: str, xlabel: str, out: Path, plt) -> None:
    n = bin_data["n"]
    keep = n > 0
    if not keep.any():
        return
    x = np.arange(len(labels))
    series = [
        ("GNN–Transformer", "sq"),
        ("Unchanged congestion delay", "sq_unchanged_delay"),
        ("No change in total travel time", "sq_no_change"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    width = 0.25
    for j, (label, key) in enumerate(series):
        rmse = np.zeros(n.shape, dtype=float)
        rmse[keep] = np.sqrt(bin_data[key][keep] / n[keep])
        ax.bar(x[keep] + (j - 1) * width, rmse[keep], width, label=label)
    ax.set_xticks(x, labels)
    ax.set(xlabel=xlabel, ylabel="RMSE of total travel-time change (s)", title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    for i in x[keep]:
        height = max(np.sqrt(bin_data[key][i] / n[i]) for _, key in series)
        ax.annotate(f"n={n[i]:,}", (i, height), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.margins(y=.13)
    fig.tight_layout()
    fig.savefig(out / name, dpi=180)
    plt.close(fig)


def _plot_scenarios(rows: list[dict], out: Path, plt) -> None:
    valid = [r for r in rows if r["road_rmse_s"] is not None]
    if not valid:
        return
    data = np.array([[r["road_rmse_s"], r["road_no_change_rmse_s"],
                      r["road_unchanged_delay_rmse_s"]] for r in valid], dtype=float)
    data = data[np.argsort(data[:, 1])]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(1, len(data) + 1)
    ax.plot(x, data[:, 0], lw=1.1, label="GNN–Transformer")
    ax.plot(x, data[:, 1], lw=1.1, label="No total TT change")
    ax.plot(x, data[:, 2], lw=1.1, label="Unchanged congestion delay")
    ax.set(xlabel="Scenario, sorted by no-total-change RMSE",
           ylabel="RMSE of total travel-time change (s)",
           title="Travel-time prediction error by scenario")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rmse_by_scenario_road.png", dpi=180)
    plt.close(fig)


def _save_plots(out: Path, checkpoint: Path, best_epoch: int, scatter: list,
                magnitude: dict, congestion: dict, rows: list[dict], metrics: dict,
                history: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_history(history or checkpoint.parent / "history.csv", out, best_epoch, plt)
    if metrics["count"]:
        _plot_scatter(scatter, metrics, out, plt)
        _plot_bins(magnitude, MAGNITUDE_LABELS, "rmse_by_change_road.png",
                   "Error by actual total travel-time change",
                   "Absolute actual total travel-time change (s)", out, plt)
        _plot_bins(congestion, BASELINE_DELAY_LABELS,
                   "rmse_by_baseline_congestion_road.png",
                   "Error by baseline congestion delay",
                   "Baseline congestion delay (s)", out, plt)
        _plot_scenarios(rows, out, plt)


# --- Parallel and resumable test-set evaluation ---
# The output format remains compatible with evaluate_fitness.py (07).
_EVAL_WORKER = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        # Supplying an open file avoids numpy appending '.npz' to a .tmp filename.
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _empty_bins(length: int) -> dict:
    return {key: np.zeros(length, dtype=np.int64 if key == 'n' else np.float64)
            for key in ('n', 'sq', 'sq_no_change', 'sq_unchanged_delay')}


def _init_eval_worker(gnn_dir: str, checkpoint: str, scenario_ids: list[str],
                      device_name: str, threads: int, output: str, plots: bool,
                      save_predictions: bool, max_scatter: int) -> None:
    """One model and shared fixed-graph dataset per process, not per scenario."""
    global _EVAL_WORKER
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # May already be initialized in a single-process caller.
        pass
    root = Path(gnn_dir)
    data = ScenarioDataset(root / 'gnn_static', root / 'gnn_inputs',
                           root / 'dynamic', root / 'targets', scenario_ids)
    with np.load(root / 'gnn_static' / 'baseline_tt.npz', allow_pickle=False) as cache:
        baseline_ff = cache['baseline_freeflow_s']
        if int(cache['hour']) != data.hour:
            raise ValueError('Baseline free-flow reference hour does not match dataset')
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device_name == 'auto' else device_name)
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if (ckpt['static_feature_names'] != data.static_feature_names
            or ckpt['dynamic_feature_names'] != data.dynamic_feature_names
            or ckpt['target_channels'] != data.target_channels):
        raise ValueError('Checkpoint feature/output names do not match dataset')
    model = GNNTransformer(**ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    _EVAL_WORKER = {
        'data': data, 'model': model, 'device': device,
        'fixed_device': {k: v.to(device) for k, v in data.fixed.items()},
        'baseline_tt': data.fixed['baseline_tt_s'].numpy(),
        'baseline_ff': baseline_ff,
        'output': Path(output), 'plots': plots,
        'save_predictions': save_predictions,
        'max_scatter': max_scatter,
    }


def _process_eval_index(index: int) -> str:
    """Evaluate one scenario; write predictions and a resumable partial record."""
    state = _EVAL_WORKER
    data = state['data']
    item = data[index]
    sid = item['scenario_id']
    device = state['device']
    dynamic = item['x_dynamic']
    truth = item['y_delta']
    valid = item['valid_mask'].numpy()
    with torch.inference_mode():
        pred = state['model'](state['fixed_device']['x_static'], dynamic.to(device),
                              state['fixed_device']['x_baseline'],
                              state['fixed_device']['edge_index'])
    if pred.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError(f'Shape mismatch in {sid}')
    pred_array = pred.cpu().numpy()
    truth_array = truth.numpy()
    reconstruction = reconstruct_road_travel_times(
        state['baseline_tt'], state['baseline_ff'], dynamic.numpy(),
        data.dynamic_feature_names, pred_array, valid,
        actual_delay_delta_s=truth_array)
    predicted = reconstruction['predicted_total_delta_s'][valid]
    actual = reconstruction['actual_total_delta_s'][valid]
    actual_delay = truth_array[valid].astype(np.float64)
    abs_tt = reconstruction['actual_tt_s'][valid]
    predicted_tt = reconstruction['predicted_tt_s'][valid]
    scenario_ff = reconstruction['scenario_freeflow_s'][valid]
    n = actual.size
    stats = _empty_stats()
    row = {
        'scenario': sid, 'road_n': int(n),
        'road_rmse_s': None, 'road_no_change_rmse_s': None,
        'road_unchanged_delay_rmse_s': None,
        'road_mae_s': None, 'road_mean_error_s': None,
        'road_predicted_below_freeflow': 0,
    }
    magnitudes = _empty_bins(6)
    congestion = _empty_bins(5)
    sample_true = np.empty(0, dtype=np.float64)
    sample_pred = np.empty(0, dtype=np.float64)
    if n:
        error = predicted - actual
        sq = np.square(error)
        stats.update({
            'n': int(n), 'abs': float(np.abs(error).sum()),
            'sq': float(sq.sum()),
            'sq_no_change': float(np.square(actual).sum()),
            'sq_unchanged_delay': float(np.square(actual_delay).sum()),
            'sum_total': float(actual.sum()),
            'sum_total2': float(np.square(actual).sum()),
            'sum_delay': float(actual_delay.sum()),
            'sum_delay2': float(np.square(actual_delay).sum()),
            'sum_absolute': float(abs_tt.sum()),
            'sum_absolute2': float(np.square(abs_tt).sum()),
            'bias': float(error.sum()),
            'below_ff': int(np.count_nonzero(predicted_tt < scenario_ff - 0.01)),
        })
        row.update({
            'road_rmse_s': float(np.sqrt(sq.mean())),
            'road_no_change_rmse_s': float(np.sqrt(np.square(actual).mean())),
            'road_unchanged_delay_rmse_s': float(np.sqrt(np.square(actual_delay).mean())),
            'road_mae_s': float(np.abs(error).mean()),
            'road_mean_error_s': float(error.mean()),
            'road_predicted_below_freeflow': stats['below_ff'],
        })
        if state['plots']:
            b = np.searchsorted(MAGNITUDE_EDGES_S, np.abs(actual), side='left')
            c = np.searchsorted(BASELINE_DELAY_EDGES_S,
                                np.maximum(0., state['baseline_tt'][valid] - state['baseline_ff'][valid]),
                                side='left')
            for bins, indices in ((magnitudes, b), (congestion, c)):
                bins['n'] += np.bincount(indices, minlength=len(bins['n']))
                bins['sq'] += np.bincount(indices, weights=sq, minlength=len(bins['n']))
                bins['sq_no_change'] += np.bincount(indices, weights=np.square(actual),
                                                   minlength=len(bins['n']))
                bins['sq_unchanged_delay'] += np.bincount(indices,
                                                           weights=np.square(actual_delay),
                                                           minlength=len(bins['n']))
            # Reproducible regardless of process scheduling and resume order.
            seed = int.from_bytes(hashlib.sha256(sid.encode('utf-8')).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            chosen = rng.choice(n, size=min(n, state['max_scatter']), replace=False)
            sample_true = actual[chosen].astype(np.float64)
            sample_pred = predicted[chosen].astype(np.float64)
    output = state['output']
    if state['save_predictions']:
        _atomic_npz(output / 'predictions' / f'{sid}.npz',
                    predicted_delay_change_s=pred_array.astype(np.float32),
                    actual_delay_change_s=truth_array.astype(np.float32),
                    freeflow_change_s=reconstruction['freeflow_change_s'].astype(np.float32),
                    predicted_total_tt_change_s=reconstruction['predicted_total_delta_s'].astype(np.float32),
                    actual_total_tt_change_s=reconstruction['actual_total_delta_s'].astype(np.float32),
                    predicted_tt_s=reconstruction['predicted_tt_s'].astype(np.float32),
                    actual_tt_s=reconstruction['actual_tt_s'].astype(np.float32),
                    valid_mask=valid)
    record = {'scenario': sid, 'row': row, 'stats': stats}
    # The partial record is written LAST; it marks successful completion.
    _atomic_npz(output / 'partial_records' / f'{sid}.npz',
                record_json=np.asarray(json.dumps(record, allow_nan=False)),
                mag_n=magnitudes['n'], mag_sq=magnitudes['sq'],
                mag_sq_no_change=magnitudes['sq_no_change'],
                mag_sq_unchanged_delay=magnitudes['sq_unchanged_delay'],
                cong_n=congestion['n'], cong_sq=congestion['sq'],
                cong_sq_no_change=congestion['sq_no_change'],
                cong_sq_unchanged_delay=congestion['sq_unchanged_delay'],
                scatter_actual=sample_true, scatter_pred=sample_pred)
    return sid


def _load_partial(path: Path, sid: str) -> tuple[dict, dict, dict, dict, tuple]:
    with np.load(path, allow_pickle=False) as z:
        record = json.loads(str(z['record_json'].item()))
        if record['scenario'] != sid or record['row']['scenario'] != sid:
            raise ValueError(f'Partial scenario ID mismatch: {path}')
        magnitude = {key: z[f'mag_{key}'].copy()
                     for key in ('n', 'sq', 'sq_no_change', 'sq_unchanged_delay')}
        congestion = {key: z[f'cong_{key}'].copy()
                      for key in ('n', 'sq', 'sq_no_change', 'sq_unchanged_delay')}
        scatter = (z['scatter_actual'].copy(), z['scatter_pred'].copy())
    return record['row'], record['stats'], magnitude, congestion, scatter


def _baseline_hour(gnn_dir: Path) -> int:
    with np.load(gnn_dir / 'gnn_static' / 'baseline_tt.npz', allow_pickle=False) as cache:
        return int(cache['hour'])


def evaluate(gnn_dir: Path, splits_file: Path, checkpoint: Path, output: Path,
             split: str = 'val', device_name: str = 'cpu', *, plots: bool = True,
             save_predictions: bool = False, history: Path | None = None,
             workers: int = 1, threads_per_worker: int = 1,
             resume: bool = False) -> dict:
    """Evaluate a frozen split with identical metrics for 1 or N Python workers.

    Per-scenario partial records and optional predictions are persisted so a
    failed/interrupted job can resume without repeating completed inference.
    """
    if workers < 1 or threads_per_worker < 1:
        raise ValueError('workers and threads_per_worker must be positive')
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and workers * threads_per_worker > int(allocated):
        raise ValueError(f'{workers} workers x {threads_per_worker} threads exceeds '
                         f'SLURM_CPUS_PER_TASK={allocated}')
    if workers > 1 and (device_name == 'cuda' or
                        (device_name == 'auto' and torch.cuda.is_available())):
        raise ValueError('Multi-process evaluation currently supports CPU inference only')
    gnn_dir, splits_file = Path(gnn_dir).resolve(), Path(splits_file).resolve()
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    ids = load_splits(splits_file)[split]
    if not ids:
        raise ValueError('Empty evaluation split')
    # Fail before launching worker processes if any targets have wrong semantics.
    for sid in ids:
        meta = json.loads((gnn_dir / 'targets' / sid / 'target_metadata.json').read_text())
        if meta.get('target_type') != TARGET_TYPE:
            raise ValueError(f'{sid}: expected target_type={TARGET_TYPE}, '
                             f'found {meta.get("target_type")!r}')
    manifest = {
        'evaluation_version': 'parallel_resumable_v1',
        'gnn_dir': str(gnn_dir), 'splits': str(splits_file),
        'splits_sha256': _sha256_file(splits_file), 'split': split,
        'scenario_ids': ids,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256_file(checkpoint),
        'target_type': TARGET_TYPE,
        'evaluator_sha256': _sha256_file(Path(__file__)),
        'plots': bool(plots),
        'save_predictions': bool(save_predictions),
        'max_scatter_per_scenario': max(1, SCATTER_MAX_POINTS // len(ids)),
    }
    config_file = output / 'evaluation_config.json'
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f'Output already exists; choose a fresh --out or '
                                  f'use --resume for the exact same evaluation: {output}')
        if not config_file.is_file():
            raise ValueError(f'Cannot resume: missing evaluation_config.json in {output}')
        if json.loads(config_file.read_text()) != manifest:
            raise ValueError('Cannot resume: checkpoint/splits/settings differ from '
                             'saved evaluation_config.json')
    else:
        if resume:
            raise ValueError('Cannot resume: evaluation output does not exist')
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_file, manifest)
    (output / 'partial_records').mkdir(exist_ok=True)
    if save_predictions:
        (output / 'predictions').mkdir(exist_ok=True)

    remaining_indices = []
    for index, sid in enumerate(ids):
        record = output / 'partial_records' / f'{sid}.npz'
        pred = output / 'predictions' / f'{sid}.npz'
        if record.is_file() and (not save_predictions or pred.is_file()):
            # Fail closed on corrupt partials instead of silently reusing them.
            _load_partial(record, sid)
        else:
            remaining_indices.append(index)
    print(f'Evaluating {len(ids)} {split} scenarios: {len(remaining_indices)} new, '
          f'{len(ids)-len(remaining_indices)} cached; workers={workers}, '
          f'threads/worker={threads_per_worker}', flush=True)
    if remaining_indices:
        args = (str(gnn_dir), str(checkpoint), ids, device_name, threads_per_worker,
                str(output), plots, save_predictions, manifest['max_scatter_per_scenario'])
        if workers == 1:
            _init_eval_worker(*args)
            for n_done, index in enumerate(remaining_indices, 1):
                _process_eval_index(index)
                if n_done % 50 == 0 or n_done == len(remaining_indices):
                    print(f'Completed {n_done}/{len(remaining_indices)} new scenarios', flush=True)
        else:
            # spawn is safer than forking a parent that has initialized PyTorch.
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_eval_worker, initargs=args) as executor:
                futures = {executor.submit(_process_eval_index, index): ids[index]
                           for index in remaining_indices}
                for n_done, future in enumerate(as_completed(futures), 1):
                    sid = futures[future]
                    try:
                        if future.result() != sid:
                            raise ValueError(f'Worker returned wrong scenario for {sid}')
                    except Exception as exc:
                        raise RuntimeError(f'GNN evaluation failed for {sid}: {exc}') from exc
                    if n_done % 50 == 0 or n_done == len(remaining_indices):
                        print(f'Completed {n_done}/{len(remaining_indices)} new scenarios', flush=True)

    # Deterministic aggregation in frozen split order, not process completion order.
    stats = _empty_stats()
    magnitude = _empty_bins(6)
    congestion = _empty_bins(5)
    rows = []
    scatter = []
    for sid in ids:
        row, piece, mag, cong, samples = _load_partial(
            output / 'partial_records' / f'{sid}.npz', sid)
        rows.append(row)
        for key in stats:
            stats[key] += piece[key]
        for key in magnitude:
            magnitude[key] += mag[key]
            congestion[key] += cong[key]
        if samples[0].size:
            scatter.append(samples)
    result = _metrics(stats)
    # Checkpoint epoch used only for metadata/plotting, load on CPU once.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    summary = {
        'split': split, 'scenarios': len(ids),
        'hour': _baseline_hour(gnn_dir),
        'best_epoch': int(ckpt['epoch']),
        'target_type': TARGET_TYPE,
        'metrics': {CHANNEL: result},
        'units': 'seconds; main metrics use reconstructed TOTAL travel-time changes',
        'benchmarks': {
            'no_change': 'T_scenario_pred = T_baseline',
            'unchanged_delay': 'T_scenario_pred = T_baseline + (FF_scenario - FF_baseline)',
        },
    }
    _atomic_json(output / 'summary.json', summary)
    csv_path = output / 'scenario_metrics.csv'
    tmp = csv_path.with_name(csv_path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, csv_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    if plots:
        _save_plots(output, checkpoint, int(ckpt['epoch']), scatter,
                    magnitude, congestion, rows, result, history)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gnn-dir', type=Path, default=PROJECT_ROOT / 'data' / 'GNN')
    parser.add_argument('--splits', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--history', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed scenarios only with identical checkpoint, split and settings')
    args = parser.parse_args()
    evaluate(args.gnn_dir, args.splits or args.gnn_dir / 'splits.json',
             args.checkpoint, args.out, args.split, args.device,
             plots=not args.no_plots, save_predictions=args.save_predictions,
             history=args.history, workers=args.workers,
             threads_per_worker=args.threads_per_worker, resume=args.resume)


if __name__ == '__main__':
    main()
