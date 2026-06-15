"""
plot_results.py – Generates performance visualisations for Project MIST.
Located in src/utils/

Produces per-algorithm charts:
    1. Time taken per lane (bar chart, colour-coded by Lane_Completed)
    2. Distance travelled per lane (bar chart)
    3. Completion rate summary (pie or bar)

CSV columns expected (written by mist_controller.py):
    Timestamp_Wall_Clock | Lane_Index | Navigation_Mode |
    Time_Taken_Seconds   | Distance_Traveled_Meters | Lane_Completed
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe for headless Webots runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Resolve paths ──────────────────────────────────────────────────────────────
SRC_UTILS = Path(__file__).resolve().parent
MIST_ROOT = SRC_UTILS.parent.parent

LOG_DIR  = MIST_ROOT / "results" / "logs"
PLOT_DIR = MIST_ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _completion_colour(completed_series: pd.Series):
    """Return a list of bar colours: green = completed, red = not."""
    return ["#4caf50" if v else "#f44336" for v in completed_series]


def generate_plots():
    log_files = list(LOG_DIR.glob("*.csv"))

    if not log_files:
        print(f"⚠️  No CSV files found in {LOG_DIR}")
        return

    all_summaries = []

    for file in log_files:
        df = pd.read_csv(file)

        # ── Column validation ──────────────────────────────────────────────────
        required = {"Lane_Index", "Time_Taken_Seconds",
                    "Distance_Traveled_Meters", "Navigation_Mode"}
        missing  = required - set(df.columns)
        if missing:
            print(f"❌ {file.name}: missing columns {missing} – skipping.")
            continue

        # Lane_Completed may be absent from older logs → default False
        if "Lane_Completed" not in df.columns:
            print(f"⚠️  {file.name}: no 'Lane_Completed' column – assuming all incomplete.")
            df["Lane_Completed"] = False

        # Normalise bool (CSV stores True/False as strings in some writers)
        df["Lane_Completed"] = df["Lane_Completed"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
        )

        mode = df["Navigation_Mode"].iloc[0].split("_STALL")[0]   # strip suffix
        print(f"\n📊 Plotting: {file.name}  (mode={mode}, rows={len(df)})")

        # ── 1. Time per lane ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        colours = _completion_colour(df["Lane_Completed"])
        ax.bar(df["Lane_Index"].astype(str), df["Time_Taken_Seconds"],
               color=colours, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Lane Index")
        ax.set_ylabel("Time Taken (seconds)")
        ax.set_title(f"{mode} – Time per Lane")
        ax.set_ylim(bottom=0)

        green_patch = mpatches.Patch(color="#4caf50", label="Completed")
        red_patch   = mpatches.Patch(color="#f44336", label="Stalled / Timeout")
        ax.legend(handles=[green_patch, red_patch])

        out = PLOT_DIR / f"{file.stem}_time.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"   ✅ {out.name}")

        # ── 2. Distance per lane ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["Lane_Index"].astype(str), df["Distance_Traveled_Meters"],
               color=colours, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Lane Index")
        ax.set_ylabel("Distance Travelled (m)")
        ax.set_title(f"{mode} – Distance per Lane")
        ax.set_ylim(bottom=0)
        ax.legend(handles=[green_patch, red_patch])

        out = PLOT_DIR / f"{file.stem}_distance.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"   ✅ {out.name}")

        # ── 3. Per-file completion summary ─────────────────────────────────────
        n_completed = df["Lane_Completed"].sum()
        n_total     = len(df)
        all_summaries.append({
            "Mode":       mode,
            "Lanes Run":  n_total,
            "Completed":  int(n_completed),
            "Stalled":    n_total - int(n_completed),
            "Rate (%)":   round(100.0 * n_completed / n_total, 1) if n_total > 0 else 0,
            "Avg Time (s)":   round(df["Time_Taken_Seconds"].mean(), 2),
            "Avg Dist (m)":   round(df["Distance_Traveled_Meters"].mean(), 2),
        })

    # ── 4. Cross-algorithm comparison (if multiple modes logged) ───────────────
    if len(all_summaries) > 1:
        summary_df = pd.DataFrame(all_summaries)
        fig, axes  = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(summary_df["Mode"], summary_df["Rate (%)"],
                    color=["#2196f3", "#ff9800", "#9c27b0"][:len(summary_df)],
                    edgecolor="black", linewidth=0.5)
        axes[0].set_ylabel("Completion Rate (%)")
        axes[0].set_title("Completion Rate by Algorithm")
        axes[0].set_ylim(0, 105)
        for bar, val in zip(axes[0].patches, summary_df["Rate (%)"]):
            axes[0].text(bar.get_x() + bar.get_width() / 2.0,
                         bar.get_height() + 1.5,
                         f"{val}%", ha="center", fontsize=10)

        axes[1].bar(summary_df["Mode"], summary_df["Avg Time (s)"],
                    color=["#2196f3", "#ff9800", "#9c27b0"][:len(summary_df)],
                    edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Avg Time per Lane (s)")
        axes[1].set_title("Average Lane Time by Algorithm")

        out = PLOT_DIR / "comparison_all_algorithms.png"
        fig.suptitle("MIST Algorithm Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n📊 Comparison chart: {out.name}")

        # Print summary table to stdout
        print("\n" + "="*60)
        print(summary_df.to_string(index=False))
        print("="*60)


if __name__ == "__main__":
    generate_plots()