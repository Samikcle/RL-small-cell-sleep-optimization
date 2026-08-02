"""
convert_rl_npy_to_csv.py

Standalone utility for converting reinforcement-learning result .npy files into CSV files.

Designed for the latest model output folders:
- DDQN:       ddqn_per_results_c_5
- PPO:        ppo_results_4
- Q-Learning: qlearning_results_c_10

What it creates:
1. csv_results/<MODEL>/<each_npy_file>.csv
   - Individual CSV conversion for each result .npy file.

2. csv_results/all_models_combined_metrics.csv
   - One episode-level table combining DDQN, PPO, Q-learning, and their baselines.

3. csv_results/conversion_log.csv
   - Record of converted, missing, or skipped files.

Run from the folder that contains:
- ddqn_per_results_c_5/
- ppo_results_4/
- qlearning_results_c_10/

Usage:
    python3 convert_rl_npy_to_csv.py

Optional:
    python3 convert_rl_npy_to_csv.py --base-dir /path/to/project
    python3 convert_rl_npy_to_csv.py --output-dir my_csv_results
    python3 convert_rl_npy_to_csv.py --include-model-artifacts

By default, very large model artifacts such as Q-tables, metadata, and per-agent checkpoint arrays
are skipped because they are not usually useful for result analysis tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. MODEL OUTPUT CONFIGURATION
# ============================================================
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "DDQN": {
        "folder": "ddqn_per_results_c_5_500_2",
        "rl_energy": "rl_energy_per_episode.npy",
        "rl_power": "total_power_per_episode.npy",
        "rl_sinr": "sinr_per_episode.npy",
        "rl_reward": "avg_reward_per_episode.npy",
        "rl_efficiency": "energy_efficiency_per_episode.npy",
        "baseline_energy": "baseline_energy_per_episode.npy",
        "baseline_power": "baseline_power_per_episode.npy",
        "baseline_sinr": "baseline_sinr_per_episode.npy",
        "baseline_efficiency": "energy_efficiency_baseline.npy",
        "loss": "loss_history.npy",
        "step_rewards": "step_rewards.npy",
    },
    "PPO": {
        "folder": "ppo_results_4_500_5",
        "rl_energy": "rl_energy_per_episode.npy",
        # This optional filename is included when PPO power history is exported.
        "rl_power": "total_power_per_episode.npy",
        "rl_sinr": "sinr_per_episode.npy",
        "rl_reward": "avg_reward_per_episode.npy",
        "rl_efficiency": "energy_efficiency_per_episode.npy",
        "baseline_energy": "baseline_energy_per_episode.npy",
        "baseline_power": "baseline_power_per_episode.npy",
        "baseline_sinr": "baseline_sinr_per_episode.npy",
        "baseline_efficiency": "energy_efficiency_baseline.npy",
        "actor_loss": "trained_ppo_agent_actor_loss.npy",
        "critic_loss": "trained_ppo_agent_critic_loss.npy",
        "total_loss": "trained_ppo_agent_total_loss.npy",
    },
    "Q-Learning": {
        "folder": "qlearning_results_c_10_500_2",
        "rl_energy": "rl_energy_per_episode.npy",
        "rl_energy_alt": "rl_energy_qlearning.npy",
        "rl_power": "total_power_per_episode.npy",
        "rl_sinr": "sinr_per_episode.npy",
        "rl_reward": "avg_reward_per_episode.npy",
        "rl_efficiency": "energy_efficiency_per_episode.npy",
        "baseline_energy": "baseline_energy_per_episode.npy",
        "baseline_energy_alt": "baseline_energy_qlearning.npy",
        "baseline_power": "baseline_power_per_episode.npy",
        "baseline_power_alt": "baseline_power_qlearning.npy",
        "baseline_sinr": "baseline_sinr_per_episode.npy",
        "baseline_sinr_alt": "baseline_sinr_qlearning.npy",
        "baseline_efficiency": "energy_efficiency_baseline.npy",
        "td_error": "trained_q_agent_td_error.npy",
    },
}

# Used only if an efficiency .npy file is missing and needs to be calculated.
NUM_UES = 30
SIM_TIME = 10
PACKET_INTERVAL = 0.02778
PACKET_SIZE_BYTES = 1400
TOTAL_BITS = (SIM_TIME / PACKET_INTERVAL) * NUM_UES * PACKET_SIZE_BYTES * 8

# Files usually not useful for analysis CSVs unless explicitly requested.
SKIP_PATTERNS = (
    "_qtable.npy",
    "_meta.npy",
)


# ============================================================
# 2. BASIC UTILITIES
# ============================================================
def load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"[ERROR] Could not load {path}: {exc}")
        return None


def find_first_existing(folder: Path, filenames: Iterable[str]) -> Optional[Path]:
    for filename in filenames:
        if not filename:
            continue
        path = folder / filename
        if path.exists():
            return path
    return None


def to_1d_float_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
        return arr.reshape(-1)
    except Exception:
        return None


def safe_get(arr: Optional[np.ndarray], index: int) -> str:
    if arr is None or index >= len(arr):
        return ""
    value = arr[index]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def energy_efficiency_from_energy(energy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if energy is None:
        return None
    return np.asarray([TOTAL_BITS / e if e > 0 else 0.0 for e in energy], dtype=float)


def should_skip_file(path: Path, include_model_artifacts: bool) -> bool:
    if include_model_artifacts:
        return False
    name = path.name
    return any(pattern in name for pattern in SKIP_PATTERNS)


# ============================================================
# 3. GENERIC .NPY TO .CSV CONVERSION
# ============================================================
def write_scalar_or_dict_csv(value: Any, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if isinstance(value, dict):
            writer.writerow(["key", "value"])
            for k, v in value.items():
                writer.writerow([k, v])
        else:
            writer.writerow(["value"])
            writer.writerow([value])


def write_array_csv(array: np.ndarray, csv_path: Path) -> None:
    """Write scalar, 1D, 2D, or higher-dimensional array into a readable CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle object scalar containing dict/meta data.
    if array.shape == ():
        item = array.item()
        write_scalar_or_dict_csv(item, csv_path)
        return

    # Convert object arrays as best as possible. If nested objects exist, store JSON strings.
    arr = np.asarray(array)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if arr.ndim == 1:
            writer.writerow(["index", "value"])
            for i, value in enumerate(arr):
                writer.writerow([i + 1, value_to_csv_cell(value)])

        elif arr.ndim == 2:
            header = ["row_index"] + [f"col_{j}" for j in range(arr.shape[1])]
            writer.writerow(header)
            for i, row in enumerate(arr):
                writer.writerow([i + 1] + [value_to_csv_cell(v) for v in row])

        else:
            writer.writerow(["flat_index", "multi_index", "value"])
            for flat_i, index_tuple in enumerate(np.ndindex(arr.shape), start=1):
                writer.writerow([flat_i, json.dumps(index_tuple), value_to_csv_cell(arr[index_tuple])])


def value_to_csv_cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        try:
            return json.dumps(np.asarray(value).tolist())
        except Exception:
            return str(value)
    return value


def convert_all_npy_in_folder(
    model_name: str,
    folder: Path,
    output_folder: Path,
    include_model_artifacts: bool,
    log_rows: List[List[str]],
) -> None:
    if not folder.exists():
        log_rows.append([model_name, str(folder), "", "missing_folder", "Folder not found"])
        return

    model_csv_folder = output_folder / model_name.replace(" ", "_").replace("-", "_")
    npy_files = sorted(folder.glob("*.npy"))

    if not npy_files:
        log_rows.append([model_name, str(folder), "", "no_npy_files", "No .npy files found"])
        return

    for npy_path in npy_files:
        if should_skip_file(npy_path, include_model_artifacts):
            log_rows.append([model_name, str(folder), npy_path.name, "skipped", "Model artifact skipped"])
            continue

        array = load_npy(npy_path)
        if array is None:
            log_rows.append([model_name, str(folder), npy_path.name, "error", "Could not load file"])
            continue

        csv_path = model_csv_folder / f"{npy_path.stem}.csv"
        try:
            write_array_csv(array, csv_path)
            log_rows.append([model_name, str(folder), npy_path.name, "converted", str(csv_path)])
        except Exception as exc:
            log_rows.append([model_name, str(folder), npy_path.name, "error", str(exc)])


# ============================================================
# 4. COMBINED EPISODE-LEVEL CSV FOR COMPARISON
# ============================================================
def load_metric(folder: Path, *filenames: str) -> Optional[np.ndarray]:
    path = find_first_existing(folder, filenames)
    if path is None:
        return None
    return to_1d_float_array(load_npy(path))


def write_combined_metrics_csv(base_dir: Path, output_dir: Path, log_rows: List[List[str]]) -> None:
    output_path = output_dir / "all_models_combined_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[List[Any]] = []
    header = [
        "model",
        "episode",
        "rl_avg_reward",
        "rl_energy_j",
        "rl_power_w",
        "rl_sinr_db",
        "rl_energy_efficiency_bits_per_j",
        "baseline_energy_j",
        "baseline_power_w",
        "baseline_sinr_db",
        "baseline_energy_efficiency_bits_per_j",
    ]

    for model_name, cfg in MODEL_CONFIGS.items():
        folder = base_dir / cfg["folder"]
        if not folder.exists():
            log_rows.append([model_name, str(folder), "combined_metrics", "missing_folder", "Folder not found"])
            continue

        rl_reward = load_metric(folder, cfg.get("rl_reward", ""))
        rl_energy = load_metric(folder, cfg.get("rl_energy", ""), cfg.get("rl_energy_alt", ""))
        rl_power = load_metric(folder, cfg.get("rl_power", ""))
        rl_sinr = load_metric(folder, cfg.get("rl_sinr", ""))
        rl_eff = load_metric(folder, cfg.get("rl_efficiency", ""))
        if rl_eff is None:
            rl_eff = energy_efficiency_from_energy(rl_energy)

        base_energy = load_metric(folder, cfg.get("baseline_energy", ""), cfg.get("baseline_energy_alt", ""))
        base_power = load_metric(folder, cfg.get("baseline_power", ""), cfg.get("baseline_power_alt", ""))
        base_sinr = load_metric(folder, cfg.get("baseline_sinr", ""), cfg.get("baseline_sinr_alt", ""))
        base_eff = load_metric(folder, cfg.get("baseline_efficiency", ""))
        if base_eff is None:
            base_eff = energy_efficiency_from_energy(base_energy)

        lengths = [
            len(a) for a in [rl_reward, rl_energy, rl_power, rl_sinr, rl_eff, base_energy, base_power, base_sinr, base_eff]
            if a is not None
        ]
        if not lengths:
            log_rows.append([model_name, str(folder), "combined_metrics", "no_metrics", "No recognized metric arrays found"])
            continue

        max_len = max(lengths)
        for i in range(max_len):
            rows.append([
                model_name,
                i + 1,
                safe_get(rl_reward, i),
                safe_get(rl_energy, i),
                safe_get(rl_power, i),
                safe_get(rl_sinr, i),
                safe_get(rl_eff, i),
                safe_get(base_energy, i),
                safe_get(base_power, i),
                safe_get(base_sinr, i),
                safe_get(base_eff, i),
            ])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    log_rows.append(["ALL", str(output_path.parent), output_path.name, "converted", f"{len(rows)} combined rows"])


def write_conversion_log(output_dir: Path, log_rows: List[List[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "conversion_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "source_folder", "source_file", "status", "message"])
        writer.writerows(log_rows)


# ============================================================
# 5. MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RL result .npy files to CSV files.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project directory containing the model result folders. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="csv_results",
        help="Folder where CSV files will be written. Default: csv_results.",
    )
    parser.add_argument(
        "--include-model-artifacts",
        action="store_true",
        help="Also convert model artifacts such as qtable/meta .npy files. Usually not needed.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_rows: List[List[str]] = []

    for model_name, cfg in MODEL_CONFIGS.items():
        folder = base_dir / cfg["folder"]
        convert_all_npy_in_folder(
            model_name=model_name,
            folder=folder,
            output_folder=output_dir,
            include_model_artifacts=args.include_model_artifacts,
            log_rows=log_rows,
        )

    write_combined_metrics_csv(base_dir, output_dir, log_rows)
    write_conversion_log(output_dir, log_rows)

    print("CSV conversion completed.")
    print(f"Output folder: {output_dir}")
    print(f"Combined comparison CSV: {output_dir / 'all_models_combined_metrics.csv'}")
    print(f"Conversion log: {output_dir / 'conversion_log.csv'}")


if __name__ == "__main__":
    main()
