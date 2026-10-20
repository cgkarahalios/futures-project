"""Reproduce saved research and measure crossover sharpness without downloads.

Run from the repository root: python scripts/audit_research_results.py
Outputs go to results/research_audit; upstream notebooks and data are untouched.
Durations use the existing panel's UTC observation index, NOT CME sessions.
"""
from pathlib import Path
import contextlib
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crossover_model import walk_forward_predictions, score_predictions

OUT = ROOT / "results" / "research_audit"


def read(name, folder="processed"):
    path = ROOT / "data" / folder / name
    try:
        return pd.read_parquet(path)
    except OSError as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        # Compatibility workaround for this environment's pyarrow 19 reader.
        return pd.read_parquet(path, engine="fastparquet")


def run_cells(name, indices, namespace):
    """Execute only reviewed calculation cells; omit downloads and writes."""
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    for i in indices:
        exec(compile("".join(notebook["cells"][i]["source"]), f"{name}:cell{i}", "exec"), namespace)


def sharpness(panel):
    rows, aligned = [], []
    for rid, g in panel.groupby("roll_id"):
        g = g.sort_values("bdays_to_fnd").reset_index(drop=True)
        for kind in ("volume", "oi"):
            col = f"next_{kind}_share"
            s = g[col]
            hits = g.index[s > .5]
            if len(hits) == 0:
                continue
            i = int(hits[0])
            day = int(g.loc[i, "bdays_to_fnd"])
            row = {"roll_id": rid, "measure": kind, "crossover_date": g.loc[i, "date"],
                   "crossover_panel_day": day, "missing_share_rows": int(s.isna().sum()),
                   "post_crossing_reversions": int((s.iloc[i+1:] <= .5).sum()),
                   "followup_observations": int(s.iloc[i+1:].notna().sum())}
            row["crossing_jump_pp"] = (s.iloc[i] - s.iloc[i-1]) * 100 if i > 0 else np.nan
            row["centered_slope_pp_per_observation"] = (
                (s.iloc[i+1] - s.iloc[i-1]) * 50 if 0 < i < len(g)-1 else np.nan)
            for lower, upper in ((.1,.9),(.2,.8),(.4,.6)):
                lo = g.index[s >= lower]
                hi = g.index[s >= upper]
                label = f"width_{int(lower*100)}_{int(upper*100)}"
                # First passages are descriptive. Do not invent sub-daily times.
                bracketed = len(lo) and len(hi) and (s.iloc[:lo[0]] < lower).any()
                row[label] = (int(g.loc[hi[0], "bdays_to_fnd"] - g.loc[lo[0], "bdays_to_fnd"])
                              if bracketed and hi[0] >= lo[0] else np.nan)
            rows.append(row)
            a = g[["roll_id", "bdays_to_fnd", col]].copy()
            a["offset"] = a.bdays_to_fnd - day
            a["measure"] = kind
            aligned.append(a.rename(columns={col:"share"}))
    return pd.DataFrame(rows), pd.concat(aligned, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read("zn_roll_panel.parquet")
    ohlcv = read("zn_ohlcv_daily.parquet", "raw")
    early = read("zn_early_indicator_panel.parquet")
    events = read("zn_crossover_events.parquet")
    for frame in (panel, ohlcv, early):
        frame["date"] = pd.to_datetime(frame["date"])
    metrics, aligned = sharpness(panel)
    metrics.to_csv(OUT / "crossover_sharpness_by_roll.csv", index=False)
    summary = metrics.groupby("measure")[["crossover_panel_day", "width_10_90", "width_20_80", "width_40_60",
                                         "crossing_jump_pp", "centered_slope_pp_per_observation"]].describe()
    summary.to_csv(OUT / "crossover_sharpness_summary.csv")

    ns = {"pd":pd, "np":np, "panel":panel.copy(), "ohlcv":ohlcv, "early":early,
          "events":events, "display": lambda *args: None}
    with contextlib.redirect_stdout(io.StringIO()):
        run_cells("07_build_standardized_roll_panel.ipynb", [3,5,7], ns)
        run_cells("08_test_roll_timing_costs.ipynb", [3,5,8,10,12], ns)
    predictions = ns["predictions"].copy()
    predictions.to_csv(OUT / "timing_predictions_reproduced.csv", index=False)
    ns["summary"].to_csv(OUT / "cost_proxy_summary_reproduced.csv")
    ns["paired_results"].to_csv(OUT / "cost_proxy_pairs_reproduced.csv", index=False)

    # Audit groupby.first(): it skips NaNs independently in each column.
    ordered = early.assign(decision_distance=(early.bdays_to_fnd + 10).abs()).sort_values(
        ["roll_id", "decision_distance", "date"])
    actual_rows = ordered.drop_duplicates("roll_id").set_index("roll_id")
    selected = ns["decision"].set_index("roll_id")
    leaks = []
    for feature in ns["FEATURES"]:
        for rid in selected.index:
            if pd.isna(actual_rows.loc[rid, feature]) and pd.notna(selected.loc[rid, feature]):
                candidates = ordered.loc[(ordered.roll_id == rid) & ordered[feature].notna()]
                source = candidates.iloc[0]
                leaks.append({"roll_id":rid, "feature":feature, "decision_date":actual_rows.loc[rid,"date"],
                              "source_date":source.date, "source_after_decision":bool(source.date > actual_rows.loc[rid,"date"])})
    pd.DataFrame(leaks, columns=["roll_id","feature","decision_date","source_date","source_after_decision"]).to_csv(
        OUT / "decision_row_mixing.csv", index=False)

    # Sensitivity: preserve entire decision rows; retain training-only imputation.
    corrected = ns["decision"].copy()
    for f in ns["FEATURES"]:
        corrected[f] = corrected.roll_id.map(actual_rows[f])
    ns["decision"] = corrected
    with contextlib.redirect_stdout(io.StringIO()):
        run_cells("08_test_roll_timing_costs.ipynb", [5,8,10,12], ns)
    ns["predictions"].to_csv(OUT / "timing_predictions_whole_rows.csv", index=False)
    ns["summary"].to_csv(OUT / "cost_proxy_summary_whole_rows.csv")

    scores = []
    for h in (1,3,5):
        forecast = walk_forward_predictions(early, horizon=h, min_train_rolls=6)
        score = score_predictions(forecast)
        score.insert(0, "horizon", h)
        scores.append(score)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(OUT / "probability_model_scores.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"volume":"#2563a6", "oi":"#cb6733"}
    for kind, title, ax in zip(("volume","oi"), ("Volume migration", "Open-interest migration"), axes[:2]):
        a = aligned[aligned.measure == kind]
        for _, g in a.groupby("roll_id"):
            ax.plot(g.offset, g.share*100, color=colors[kind], alpha=.12, lw=.7)
        q = a.groupby("offset").share.quantile([.25,.5,.75]).unstack()*100
        ax.fill_between(q.index, q[.25], q[.75], color=colors[kind], alpha=.18)
        ax.plot(q.index, q[.5], color=colors[kind], lw=2, label="Median; band = IQR")
        ax.axhline(50, color="gray", ls="--", lw=.8)
        ax.axvline(0, color="gray", ls="--", lw=.8)
        ax.set(xlim=(-8,5), ylim=(0,100), title=title,
               xlabel="UTC panel observations from own first >50%", ylabel="Next-contract share (%)")
        ax.legend(fontsize=8)
    for j, kind in enumerate(("volume","oi")):
        vals = metrics.loc[metrics.measure == kind, "width_20_80"].dropna()
        axes[2].scatter(np.full(len(vals),j) + np.linspace(-.12,.12,len(vals)), vals,
                        color=colors[kind], alpha=.55, s=24)
        axes[2].plot([j-.22,j+.22],[vals.median()]*2, color="black", lw=2)
    axes[2].set(xticks=[0,1], xticklabels=["Volume","Open interest"],
                title="20% to 80% first-passage width", ylabel="UTC panel observation intervals")
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle("ZN crossover sharpness | 27 rolls, 2020-2026 | provisional UTC-day analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "crossover_sharpness.png", dpi=180)
    plt.close(fig)

    stats = {
        "rolls":int(panel.roll_id.nunique()), "panel_rows":len(panel),
        "sunday_rows":int((panel.date.dt.dayofweek == 6).sum()),
        "raw_start":str(ohlcv.date.min()), "raw_end":str(ohlcv.date.max()),
        "timing_oos_rolls":len(predictions),
        "timing_model_mae":float(predictions.prediction_error_bdays.abs().mean()),
        "timing_historical_mae":float(predictions.historical_median_error_bdays.abs().mean()),
        "whole_row_model_mae":float(ns["predictions"].prediction_error_bdays.abs().mean()),
        "mixed_feature_values":len(leaks),
        "future_mixed_feature_values":sum(x["source_after_decision"] for x in leaks),
    }
    (OUT / "audit_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(summary.to_string())
    print("Reversions:", metrics.groupby("measure").post_crossing_reversions.agg(["sum","max"]).to_dict())
    print(scores.to_string(index=False))
    print(f"Saved audit outputs to {OUT}")


if __name__ == "__main__":
    main()
