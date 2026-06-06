"""
plot_results.py – Generates performance visualizations for MIST.
Now located in src/utils/
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def generate_plots():
    # Since we are in src/utils/, we go up 3 levels to reach MIST root
    SRC_UTILS = Path(__file__).resolve().parent
    MIST_ROOT = SRC_UTILS.parent.parent

    LOG_DIR = MIST_ROOT / "results" / "logs"
    PLOT_DIR = MIST_ROOT / "results" / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # List all log files
    log_files = list(LOG_DIR.glob("*.csv"))

    if not log_files:
        print(f"⚠️ No CSV files found in {LOG_DIR}")
        return

    for file in log_files:
        df = pd.read_csv(file)

        # Simple plot: Time Taken per Lane
        plt.figure(figsize=(8, 5))
        # Ensure we are plotting the right column name based on your CSV header
        if 'Time_Seconds' in df.columns:
            plt.bar(df['Lane_Index'], df['Time_Seconds'])
            plt.xlabel('Lane Index')
            plt.ylabel('Seconds')
            plt.title(f'Performance: {file.stem}')

            output_plot = PLOT_DIR / f"{file.stem}.png"
            plt.savefig(output_plot)
            print(f"✅ Generated plot for {file.stem} at {output_plot}")
        else:
            print(f"❌ Could not find 'Time_Seconds' in {file.name}")


if __name__ == "__main__":
    generate_plots()