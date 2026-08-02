# ============================================================
# POST-PROCESSING SCRIPT:
# Add SINR quality threshold lines and percentage-based graphs
#
# This script does NOT rerun ns-3 and does NOT reload trained models.
# It only uses already saved .npy result files from:
#   all_model_test_results/
#
# The baseline is plotted directly, so no additional horizontal baseline line is added.
#
# Expected input structure:
#   all_model_test_results/ddqn/sinr.npy
#   all_model_test_results/ppo/sinr.npy
#   all_model_test_results/q_learning/sinr.npy
#   all_model_test_results/baseline/sinr.npy
#   all_model_test_results/<model>/energy.npy
#   all_model_test_results/<model>/efficiency.npy
#
# Outputs are saved to:
#   all_model_test_results/threshold_percentage_plots/
# ============================================================

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================
DEFAULT_INPUT_DIR = "all_model_test_results"
DEFAULT_OUTPUT_SUBDIR = "threshold_percentage_plots"

MODELS = {
    "DDQN": "ddqn",
    "PPO": "ppo",
    "Q-Learning": "q_learning",
    "Baseline": "baseline",
}

SINR_THRESHOLDS = {
    "Poor/Fair boundary": 0.0,
    "Fair/Good boundary": 13.0,
    "Good/Excellent boundary": 20.0,
}

# Dark magenta is used for SINR threshold lines because the model lines
# commonly use blue, orange, green, and red.
THRESHOLD_COLOR = "#8B008B"

SINR_BANDS = [
    ("Poor", -50.0, 0.0),
    ("Fair", 0.0, 13.0),
    ("Good", 13.0, 20.0),
    ("Excellent", 20.0, 50.0),
]


# ============================================================
# 2. LOADING HELPERS
# ============================================================
def load_metric(input_dir, model_folder, metric_name):
    path = Path(input_dir) / model_folder / f"{metric_name}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            f"Run test_all_models_checkpoint10.py first, or check your folder name."
        )
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)


def load_all_results(input_dir):
    results = {}

    for display_name, folder_name in MODELS.items():
        results[display_name] = {
            "sinr": load_metric(input_dir, folder_name, "sinr"),
            "energy": load_metric(input_dir, folder_name, "energy"),
            "efficiency": load_metric(input_dir, folder_name, "efficiency"),
        }

    lengths = {
        model: len(metrics["sinr"])
        for model, metrics in results.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Result arrays do not have the same length:\n"
            + "\n".join([f"{m}: {n}" for m, n in lengths.items()])
        )

    return results


def safe_percentage(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > 1e-12,
    ) * 100.0


# ============================================================
# 3. SINR THRESHOLD PLOTS
# ============================================================
def add_sinr_threshold_lines(ax, x_max):
    """
    Adds threshold lines:
      Poor      < 0
      Fair      >= 0 and < 13
      Good      >= 13 and < 20
      Excellent >= 20
    """
    for label, y_value in SINR_THRESHOLDS.items():
        ax.axhline(y=y_value, linestyle="--", linewidth=1.5, color=THRESHOLD_COLOR)
        ax.text(
            x_max + 0.3,
            y_value,
            f"{y_value:g} dB",
            va="center",
            fontsize=9,
            color=THRESHOLD_COLOR,
        )

    band_label_x = x_max + 0.8
    for band_name, y_min, y_max in SINR_BANDS:
        y_mid = (y_min + y_max) / 2.0
        if band_name == "Poor":
            y_mid = -2.0
        elif band_name == "Excellent":
            y_mid = 22.0

        ax.text(
            band_label_x,
            y_mid,
            band_name,
            va="center",
            fontsize=9,
            fontweight="bold",
        )


def plot_sinr_with_thresholds(results, output_dir):
    x = np.arange(1, len(next(iter(results.values()))["sinr"]) + 1)

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    for model_name, metrics in results.items():
        ax.plot(
            x,
            metrics["sinr"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    add_sinr_threshold_lines(ax, x_max=len(x))

    ax.set_xlabel("Test seed index")
    ax.set_ylabel("Average SINR (dB)")
    ax.set_title("SINR comparison")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_xlim(1, len(x) + 4)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sinr_comparison_with_quality_thresholds.png", dpi=300)
    plt.close()


def plot_sinr_average_bar_with_thresholds(results, output_dir):
    model_names = list(results.keys())
    avg_sinr = [np.mean(results[m]["sinr"]) for m in model_names]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(model_names, avg_sinr)

    for label, y_value in SINR_THRESHOLDS.items():
        ax.axhline(y=y_value, linestyle="--", linewidth=1.5, color=THRESHOLD_COLOR)
        ax.text(
            len(model_names) - 0.45,
            y_value,
            f"{y_value:g} dB",
            va="bottom",
            fontsize=9,
            color=THRESHOLD_COLOR,
        )

    ax.set_ylabel("Average SINR (dB)")
    ax.set_title("Average SINR by model with quality thresholds")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "average_sinr_with_quality_thresholds.png", dpi=300)
    plt.close()


# ============================================================
# 4. PERCENTAGE CONVERSION PLOTS
# ============================================================
def compute_percentage_results(results):
    """
    Converts energy and energy efficiency to percentages relative to baseline.

    Energy consumption percentage:
      model_energy_percent = model_energy / baseline_energy * 100
      Lower is better.
      Baseline appears as one plotted model at about 100%.

    Energy saving percentage:
      model_energy_saving_percent = (baseline_energy - model_energy) / baseline_energy * 100
      Higher is better.
      Baseline appears as one plotted model at about 0%.

    Energy efficiency percentage:
      model_efficiency_percent = model_efficiency / baseline_efficiency * 100
      Higher is better.
      Baseline appears as one plotted model at about 100%.
    """
    baseline_energy = results["Baseline"]["energy"]
    baseline_efficiency = results["Baseline"]["efficiency"]

    percentage = {}

    for model_name, metrics in results.items():
        energy_percent = safe_percentage(metrics["energy"], baseline_energy)
        energy_saving_percent = 100.0 - energy_percent
        efficiency_percent = safe_percentage(metrics["efficiency"], baseline_efficiency)

        percentage[model_name] = {
            "energy_percent": energy_percent,
            "energy_saving_percent": energy_saving_percent,
            "efficiency_percent": efficiency_percent,
        }

    return percentage


def plot_percentage_metric(percentage_results, metric, ylabel, title, filename, output_dir):
    x = np.arange(1, len(next(iter(percentage_results.values()))[metric]) + 1)

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    for model_name, metrics in percentage_results.items():
        ax.plot(
            x,
            metrics[metric],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    # No additional 100% or 0% reference line is added here.
    # The Baseline model itself is already plotted as one of the lines.

    ax.set_xlabel("Test seed index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()


def plot_average_percentage_bar(percentage_results, output_dir):
    model_names = list(percentage_results.keys())

    avg_energy_percent = [
        np.mean(percentage_results[m]["energy_percent"])
        for m in model_names
    ]
    avg_energy_saving_percent = [
        np.mean(percentage_results[m]["energy_saving_percent"])
        for m in model_names
    ]
    avg_efficiency_percent = [
        np.mean(percentage_results[m]["efficiency_percent"])
        for m in model_names
    ]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.bar(x - width, avg_energy_percent, width, label="Energy consumption (% of baseline)")
    ax.bar(x, avg_energy_saving_percent, width, label="Energy saving (% vs baseline)")
    ax.bar(x + width, avg_efficiency_percent, width, label="Energy efficiency (% of baseline)")

    # No additional horizontal baseline reference lines are added here.

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Average energy and efficiency percentage comparison")
    ax.grid(True, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "average_energy_efficiency_percentage_summary.png", dpi=300)
    plt.close()


# ============================================================
# 5. SAVE CSV SUMMARY
# ============================================================
def save_percentage_csv(percentage_results, output_dir):
    output_path = Path(output_dir) / "percentage_summary.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "test_seed_index",
            "energy_consumption_percent_of_baseline",
            "energy_saving_percent_vs_baseline",
            "energy_efficiency_percent_of_baseline",
        ])

        for model_name, metrics in percentage_results.items():
            n = len(metrics["energy_percent"])
            for i in range(n):
                writer.writerow([
                    model_name,
                    i + 1,
                    metrics["energy_percent"][i],
                    metrics["energy_saving_percent"][i],
                    metrics["efficiency_percent"][i],
                ])


def save_average_percentage_csv(percentage_results, output_dir):
    output_path = Path(output_dir) / "average_percentage_summary.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "avg_energy_consumption_percent_of_baseline",
            "avg_energy_saving_percent_vs_baseline",
            "avg_energy_efficiency_percent_of_baseline",
        ])

        for model_name, metrics in percentage_results.items():
            writer.writerow([
                model_name,
                np.mean(metrics["energy_percent"]),
                np.mean(metrics["energy_saving_percent"]),
                np.mean(metrics["efficiency_percent"]),
            ])


# ============================================================
# 6. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Folder containing already generated model result folders. Default: all_model_test_results",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subfolder name for new plots. Default: threshold_percentage_plots_magenta_threshold",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = input_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Loading existing result files from: {input_dir}")
    results = load_all_results(input_dir)

    print(">>> Generating SINR graphs with quality threshold lines...")
    plot_sinr_with_thresholds(results, output_dir)
    plot_sinr_average_bar_with_thresholds(results, output_dir)

    print(">>> Converting energy and efficiency values to percentages relative to baseline...")
    percentage_results = compute_percentage_results(results)

    print(">>> Generating percentage graphs without extra baseline reference lines...")
    plot_percentage_metric(
        percentage_results,
        metric="energy_percent",
        ylabel="Energy consumption (% of baseline)",
        title="Energy consumption as percentage of baseline",
        filename="energy_consumption_percent_of_baseline.png",
        output_dir=output_dir,
    )

    plot_percentage_metric(
        percentage_results,
        metric="energy_saving_percent",
        ylabel="Energy saving (% vs baseline)",
        title="Energy saving percentage compared with baseline",
        filename="energy_saving_percent_vs_baseline.png",
        output_dir=output_dir,
    )

    plot_percentage_metric(
        percentage_results,
        metric="efficiency_percent",
        ylabel="Energy efficiency (% of baseline)",
        title="Energy efficiency as percentage of baseline",
        filename="energy_efficiency_percent_of_baseline.png",
        output_dir=output_dir,
    )

    plot_average_percentage_bar(percentage_results, output_dir)

    print(">>> Saving percentage CSV summaries...")
    save_percentage_csv(percentage_results, output_dir)
    save_average_percentage_csv(percentage_results, output_dir)

    print(f"\nDone. New graphs saved in: {output_dir}")
    print("Created files:")
    for path in sorted(output_dir.glob("*")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
