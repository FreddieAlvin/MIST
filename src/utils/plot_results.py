"""
plot_results.py – Performance + fog-robustness visualisation for Project MIST.
Located in src/utils/

Reads the organised logs written by mist_controller.py:
    results/logs/<run>/fog_<level>/<MODE>_performance_log.csv
    results/logs/ppo_monitor.monitor.csv          (PPO training curve)

Produces (nothing overwrites across runs / fog levels):
  results/plots/robustness/   completion_vs_fog.png, hits_vs_fog.png,
                              time_vs_fog.png, ppo_baseline_vs_robust.png,
                              robustness_table.csv
  results/plots/<run>/fog_<level>/<MODE>/   per-lane charts + summary.csv
  results/plots/<run>/fog_<level>/comparison/   cross-algorithm charts
  results/plots/PPO/learning_curve.png      (training curve, fog-independent)

Backwards compatible: old flat logs results/logs/*.csv → run='baseline', fog=0.0.
"""
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SRC_UTILS  = Path(__file__).resolve().parent
MIST_ROOT  = SRC_UTILS.parent.parent
LOG_DIR    = MIST_ROOT / "results" / "logs"
PLOT_DIR   = MIST_ROOT / "results" / "plots"
ROBUST_DIR = PLOT_DIR / "robustness"

MODE_COLOURS = {"PPO": "#2196f3", "PURE_APF": "#ff9800", "PURE_DSTAR": "#9c27b0"}
RUN_COLOURS  = {"baseline": "#2196f3", "robust_static": "#e91e63",
                "robust_dynamic": "#00897b", "ppo_robust": "#e91e63",
                "ppo_robust_dynamic": "#00897b"}
COMPLETED_C, STALLED_C = "#4caf50", "#f44336"


def _clean_bool(s):
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))

def _fog_from_path(p):
    m = re.search(r"fog_([0-9.]+)", str(p))
    return float(m.group(1)) if m else 0.0


def load_records():
    records = []
    for f in LOG_DIR.rglob("*_performance_log.csv"):
        if "monitor" in f.name.lower():
            continue
        if not f.parent.name.startswith("fog_"):
            # Ignore stray/old flat logs — only count the organised fog_* runs.
            print(f"↪︎  ignored (not in a fog_ folder): {f.relative_to(LOG_DIR)}")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"❌ {f}: {e}"); continue
        if len(df) == 0:
            print(f"↪︎  ignored (empty / no data rows): {f.relative_to(LOG_DIR)}")
            continue
        need = {"Lane_Index", "Time_Taken_Seconds",
                "Distance_Traveled_Meters", "Navigation_Mode"}
        if need - set(df.columns):
            print(f"❌ {f.name}: missing {need - set(df.columns)} – skipping."); continue
        if "Lane_Completed" not in df: df["Lane_Completed"] = False
        if "Obstacle_Hits" not in df:  df["Obstacle_Hits"] = 0
        df["Lane_Completed"] = _clean_bool(df["Lane_Completed"])
        df["Obstacle_Hits"]  = pd.to_numeric(df["Obstacle_Hits"], errors="coerce").fillna(0)
        mode = f.stem.replace("_performance_log", "")
        run  = str(df["Run_Label"].iloc[0]) if "Run_Label" in df.columns else \
               (f.parent.parent.name if f.parent.name.startswith("fog_") else "baseline")
        # Unify the duplicated dynamic-run label so it shows as ONE series.
        if run in ("ppo_robust_dynamic", "robust_dynamic"):
            run = "robust_dynamic"
        fog  = float(df["Fog_Level"].iloc[0]) if "Fog_Level" in df.columns else _fog_from_path(f)
        records.append({"run": run, "fog": fog, "mode": mode, "df": df, "file": f})
        print(f"📥 {run} | fog={fog:.2f} | {mode:11s} | {len(df)} rows")
    return records


def per_lane(df):
    g = df.groupby("Lane_Index")
    return pd.DataFrame({
        "Time_s":     g["Time_Taken_Seconds"].mean(),
        "Dist_m":     g["Distance_Traveled_Meters"].mean(),
        "Hits":       g["Obstacle_Hits"].mean(),
        "Completion": g["Lane_Completed"].mean() * 100.0,
    }).reset_index().sort_values("Lane_Index")


def _agg(df):
    done = df["Lane_Completed"]
    return {"Completion": 100.0 * done.mean(),
            "Hits": df["Obstacle_Hits"].mean(),
            "Time": df.loc[done, "Time_Taken_Seconds"].mean() if done.any() else np.nan,
            "Dist": df["Distance_Traveled_Meters"].mean()}


def _bar(ax, x, y, colours, ylabel, title, ymax=None):
    ax.bar([str(v) for v in x], y, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Lane Index"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(bottom=0, top=ymax)
    for i, v in enumerate(y):
        ax.text(i, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)


def plot_algorithm(folder, mode, df):
    folder.mkdir(parents=True, exist_ok=True)
    lane = per_lane(df)
    cc = [COMPLETED_C if c >= 50 else STALLED_C for c in lane["Completion"]]
    for col, ylabel, fname, ymax in [
        ("Time_s",     "Time per Lane (s)",         "time_per_lane.png",       None),
        ("Dist_m",     "Distance per Lane (m)",      "distance_per_lane.png",   None),
        ("Hits",       "Obstacle Hits per Lane",     "hits_per_lane.png",       None),
        ("Completion", "Completion Rate per Lane %", "completion_per_lane.png", 105)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        _bar(ax, lane["Lane_Index"], lane[col].values, cc, ylabel, f"{mode} – {ylabel}", ymax)
        fig.tight_layout(); fig.savefig(folder / fname, dpi=150); plt.close(fig)
    lane.to_csv(folder / "summary.csv", index=False)


def plot_comparison(folder, mode_dfs):
    folder.mkdir(parents=True, exist_ok=True)
    modes = list(mode_dfs)
    summ = pd.DataFrame([{**_agg(df), "Mode": m} for m, df in mode_dfs.items()]).set_index("Mode")
    summ.to_csv(folder / "summary_table.csv")
    cols = [MODE_COLOURS.get(m, "#607d8b") for m in modes]
    for col, ylabel, fname, ymax in [
        ("Completion", "Completion Rate (%)",       "completion_rate.png",    105),
        ("Time",       "Avg Time on Completed (s)", "avg_time_completed.png", None),
        ("Dist",       "Avg Distance (m)",          "avg_distance.png",       None),
        ("Hits",       "Avg Obstacle Hits",         "avg_hits.png",           None)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        vals = summ[col].fillna(0).values
        ax.bar(summ.index, vals, color=cols, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(ylabel); ax.set_title(ylabel); ax.set_ylim(bottom=0, top=ymax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
        fig.tight_layout(); fig.savefig(folder / fname, dpi=150); plt.close(fig)
    all_lanes = sorted({int(l) for df in mode_dfs.values() for l in df["Lane_Index"]})
    for col, ylabel, fname, ymax in [
        ("Completion", "Completion (%)", "grouped_completion_per_lane.png", 105),
        ("Time_s",     "Time (s)",       "grouped_time_per_lane.png",       None),
        ("Hits",       "Obstacle Hits",  "grouped_hits_per_lane.png",       None)]:
        fig, ax = plt.subplots(figsize=(10, 5))
        n = len(modes); w = 0.8 / max(n, 1); x = np.arange(len(all_lanes))
        for i, m in enumerate(modes):
            d = per_lane(mode_dfs[m]).set_index("Lane_Index")
            y = [d[col].get(l, 0) for l in all_lanes]
            ax.bar(x + i * w, y, w, label=m, color=MODE_COLOURS.get(m, "#607d8b"),
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + w * (n - 1) / 2)
        ax.set_xticklabels([f"Lane {l}" for l in all_lanes])
        ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} per Lane (by algorithm)")
        ax.set_ylim(bottom=0, top=ymax); ax.legend()
        fig.tight_layout(); fig.savefig(folder / fname, dpi=150); plt.close(fig)


def plot_robustness(records):
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    tidy = pd.DataFrame([{"run": r["run"], "mode": r["mode"], "fog": r["fog"], **_agg(r["df"])}
                         for r in records]).sort_values(["run", "mode", "fog"])
    # After unifying duplicate dynamic labels, collapse any repeated cells.
    tidy = tidy.drop_duplicates(subset=["run", "mode", "fog"], keep="last")
    tidy.to_csv(ROBUST_DIR / "robustness_table.csv", index=False)
    print("\n" + "=" * 70 + "\n" + tidy.to_string(index=False) + "\n" + "=" * 70)
    main_run = tidy.groupby("run")["mode"].nunique().idxmax()
    main = tidy[tidy["run"] == main_run]
    for metric, ylabel, fname, ymax in [
        ("Completion", "Completion Rate (%)",       "completion_vs_fog.png", 105),
        ("Hits",       "Avg Obstacle Hits",         "hits_vs_fog.png",       None),
        ("Time",       "Avg Time on Completed (s)", "time_vs_fog.png",       None)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in sorted(main["mode"].unique()):
            d = main[main["mode"] == mode].sort_values("fog")
            ax.plot(d["fog"], d[metric], marker="o", linewidth=2,
                    color=MODE_COLOURS.get(mode, "#607d8b"), label=mode)
        ax.set_xlabel("Fog level (Lidar noise σ, fraction of max range)")
        ax.set_ylabel(ylabel); ax.set_title(f"Robustness to fog – {ylabel}  (run: {main_run})")
        ax.set_ylim(bottom=0, top=ymax); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(ROBUST_DIR / fname, dpi=150); plt.close(fig)
    print(f"   ✅ robustness curves → {ROBUST_DIR} (main run: {main_run})")
    ppo = tidy[tidy["mode"] == "PPO"]
    if ppo["run"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for run in sorted(ppo["run"].unique()):
            d = ppo[ppo["run"] == run].sort_values("fog")
            ax.plot(d["fog"], d["Completion"], marker="o", linewidth=2,
                    color=RUN_COLOURS.get(run, "#607d8b"), label=f"PPO ({run})")
        ax.set_xlabel("Fog level (Lidar noise σ)"); ax.set_ylabel("Completion Rate (%)")
        ax.set_title("PPO – clean-trained vs fog-trained robustness")
        ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(ROBUST_DIR / "ppo_baseline_vs_robust.png", dpi=150); plt.close(fig)
        print(f"   ✅ PPO baseline-vs-robust → {ROBUST_DIR}")


def plot_ppo_learning_curve():
    cands = sorted(LOG_DIR.rglob("*monitor*.csv"))
    if not cands:
        return
    folder = PLOT_DIR / "PPO"; folder.mkdir(parents=True, exist_ok=True)
    for mon in cands:
        try:
            df = pd.read_csv(mon, skiprows=1)
        except Exception:
            continue
        if "r" not in df.columns or not len(df):
            continue
        stem = mon.name.replace(".monitor.csv", "").replace("_monitor", "")
        out  = "learning_curve.png" if len(cands) == 1 else f"learning_curve_{stem}.png"
        ep = np.arange(1, len(df) + 1); roll = max(1, len(df) // 20)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, df["r"], color="#90caf9", alpha=0.4)
        ax1.plot(ep, df["r"].rolling(roll, min_periods=1).mean(), color="#1565c0",
                 linewidth=2, label=f"reward (avg {roll})")
        ax1.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Episode reward", color="#1565c0")
        ax2 = ax1.twinx()
        ax2.plot(ep, df["l"].rolling(roll, min_periods=1).mean(), color="#ef6c00",
                 linewidth=1.5, label="episode length (avg)")
        ax2.set_ylabel("Episode length", color="#ef6c00")
        ax1.set_title(f"PPO – Training Learning Curve ({stem})")
        fig.tight_layout(); fig.savefig(folder / out, dpi=150); plt.close(fig)
        print(f"   ✅ PPO learning curve → {folder / out}")


REPORT_DIR = PLOT_DIR / "report"
FOG_COLOURS = {0.0: "#90caf9", 0.02: "#1976d2", 0.05: "#0d2c54"}


def _lane_col(records, run, mode, fog, col):
    """Return per-lane Series (indexed by lane) for one (run, mode, fog), or None."""
    for r in records:
        if r["run"] == run and r["mode"] == mode and abs(r["fog"] - fog) < 1e-9:
            return per_lane(r["df"]).set_index("Lane_Index")[col]
    return None


# ── (1) Completion per lane, per model, across fog levels ─────────────────────
def fig_completion_per_lane_by_fog(records, run="baseline"):
    rec = [r for r in records if r["run"] == run]
    if not rec:
        return
    modes = sorted({r["mode"] for r in rec})
    fogs  = sorted({r["fog"] for r in rec})
    lanes = sorted({int(l) for r in rec for l in r["df"]["Lane_Index"]})
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    w = 0.8 / max(len(fogs), 1)
    x = np.arange(len(lanes))
    for ax, mode in zip(axes, modes):
        for i, fog in enumerate(fogs):
            s = _lane_col(records, run, mode, fog, "Completion")
            y = [(s.get(l, 0) if s is not None else 0) for l in lanes]
            ax.bar(x + i * w, y, w, label=f"fog {fog:.2f}",
                   color=FOG_COLOURS.get(fog, "#607d8b"), edgecolor="black", linewidth=0.4)
        ax.set_title(mode); ax.set_xlabel("Lane")
        ax.set_xticks(x + w * (len(fogs) - 1) / 2)
        ax.set_xticklabels([f"L{l}" for l in lanes])
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Lane completed (100 = yes)")
    axes[-1].legend(title="Fog level")
    fig.suptitle(f"Completion per lane, by algorithm, across fog ({run})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "completion_per_lane_by_fog.png", dpi=150); plt.close(fig)
    print("   ✅ report/completion_per_lane_by_fog.png")


# ── (2) Both PPO learning curves on one axis ──────────────────────────────────
def fig_learning_curves_combined():
    mons = sorted(LOG_DIR.rglob("*monitor*.csv"))
    if not mons:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for mon in mons:
        try:
            df = pd.read_csv(mon, skiprows=1)
        except Exception:
            continue
        if "r" not in df.columns or "l" not in df.columns or not len(df):
            continue
        steps = df["l"].cumsum()
        roll  = max(1, len(df) // 20)
        is_fog = "robust" in mon.name.lower()
        label  = "PPO trained WITH fog" if is_fog else "PPO trained normally (clean)"
        colour = "#e91e63" if is_fog else "#1976d2"
        ax.plot(steps, df["r"].rolling(roll, min_periods=1).mean(),
                color=colour, linewidth=2.2, label=label)
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Training timesteps"); ax.set_ylabel("Episode reward (smoothed)")
    ax.set_title("PPO Learning Curves — clean vs fog-trained")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "learning_curves_combined.png", dpi=150); plt.close(fig)
    print("   ✅ report/learning_curves_combined.png")


# ── (3) Per-lane performance of the 4 models (one fog level) ──────────────────
FOUR_MODELS = [  # (label, run, mode)
    ("PPO (clean)",       "baseline",   "PPO"),
    ("PPO (fog-trained)", "ppo_robust", "PPO"),
    ("APF",               "baseline",   "PURE_APF"),
    ("D* Lite",           "baseline",   "PURE_DSTAR"),
]
FOUR_COLOURS = ["#2196f3", "#e91e63", "#ff9800", "#9c27b0"]


def _fig_four_models(records, fog, col, ylabel, fname, ymax=None):
    lanes = sorted({int(l) for r in records for l in r["df"]["Lane_Index"]}) or [0, 1, 2, 3]
    x = np.arange(len(lanes)); n = len(FOUR_MODELS); w = 0.8 / n
    any_data = False
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, run, mode) in enumerate(FOUR_MODELS):
        s = _lane_col(records, run, mode, fog, col)
        if s is not None:
            any_data = True
        y = [(s.get(l, 0) if s is not None else 0) for l in lanes]
        ax.bar(x + i * w, y, w, label=label, color=FOUR_COLOURS[i],
               edgecolor="black", linewidth=0.4)
    if not any_data:
        plt.close(fig); return
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels([f"Lane {l}" for l in lanes])
    ax.set_ylabel(ylabel); ax.set_ylim(bottom=0, top=ymax)
    ax.set_title(f"{ylabel} per lane — 4 models (fog {fog:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / fname, dpi=150); plt.close(fig)
    print(f"   ✅ report/{fname}")


# ── (4) MUST-HAVE: clean headline robustness as grouped bars ──────────────────
def fig_completion_vs_fog_bars(records, run="baseline"):
    rec = [r for r in records if r["run"] == run]
    if not rec:
        return
    modes = sorted({r["mode"] for r in rec})
    fogs  = sorted({r["fog"] for r in rec})
    x = np.arange(len(fogs)); n = len(modes); w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mode in enumerate(modes):
        y = []
        for fog in fogs:
            r = next((rr for rr in rec if rr["mode"] == mode and abs(rr["fog"] - fog) < 1e-9), None)
            y.append(_agg(r["df"])["Completion"] if r else 0)
        ax.bar(x + i * w, y, w, label=mode, color=MODE_COLOURS.get(mode, "#607d8b"),
               edgecolor="black", linewidth=0.4)
        for j, v in enumerate(y):
            ax.text(x[j] + i * w, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels([f"σ={f:.2f}" for f in fogs])
    ax.set_xlabel("Fog level (Lidar noise σ)"); ax.set_ylabel("Completion Rate (%)")
    ax.set_ylim(0, 105)
    is_dyn = "dynamic" in run.lower()
    title = "Completion rate vs fog — DYNAMIC (Lane 3 moving barriers)" if is_dyn \
            else f"Completion rate vs fog ({run})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fname = f"completion_vs_fog_bars_{run}.png"
    fig.savefig(REPORT_DIR / fname, dpi=150); plt.close(fig)
    print(f"   ✅ report/{fname}")


def plot_report_figures(records):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n— Curated report figures —")
    fig_completion_per_lane_by_fog(records, run="baseline")          # (1)
    fig_learning_curves_combined()                                   # (2)
    _fig_four_models(records, 0.0, "Completion", "Completion (100 = done)",
                     "four_models_completion_per_lane_fog0.png", ymax=105)   # (3a)
    _fig_four_models(records, 0.0, "Dist_m", "Distance reached (m)",
                     "four_models_distance_per_lane_fog0.png")               # (3b) must-have
    # (4) Headline completion-vs-fog bars for EVERY run (baseline, dynamic, robust…)
    for run in sorted({r["run"] for r in records}):
        fig_completion_vs_fog_bars(records, run=run)


def generate_plots():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ppo_learning_curve()
    records = load_records()
    if not records:
        print(f"⚠️  No performance logs under {LOG_DIR}"); return
    groups = {}
    for r in records:
        groups.setdefault((r["run"], r["fog"]), {})[r["mode"]] = r["df"]
    print("\n— Detail plots per run / fog —")
    for (run, fog), mode_dfs in sorted(groups.items()):
        base = PLOT_DIR / run / f"fog_{fog:.2f}"
        for mode, df in mode_dfs.items():
            plot_algorithm(base / mode, mode, df)
        if len(mode_dfs) > 1:
            plot_comparison(base / "comparison", mode_dfs)
        print(f"   ✅ {run}/fog_{fog:.2f}  ({', '.join(mode_dfs)})")
    print("\n— Robustness curves —")
    plot_robustness(records)
    plot_report_figures(records)
    print("\n✅ Done.")


if __name__ == "__main__":
    generate_plots()