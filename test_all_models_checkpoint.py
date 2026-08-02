# ============================================================
# TEST ALL MODELS AND BASELINE WITH CHECKPOINTING

# Checkpoint behavior:
#   - Saves all completed evaluation results to:
#       all_model_test_results/evaluation_checkpoint.pkl
#   - Saves automatically every --checkpoint-interval completed tests
#     default: 10
#   - Also saves on Ctrl+C / KeyboardInterrupt and on unexpected errors
#   - Resume is enabled by default; use --restart to start from zero
# ============================================================

"""
Evaluate DDQN, PPO, Q-learning, and an always-ACTIVE baseline on the same ns-3 seeds.

Place this script in the same directory as:
  - fast_ddqn_multiagent_c_5.py
  - ppo_learning_4.py
  - tabular_q_learning_c_10.py

Run after training all three models:
  python3 test_all_models_and_baseline.py

Outputs are saved to:
  all_model_test_results/
"""

import os
import csv
import time
import random
import argparse
import pickle
import importlib.util
from pathlib import Path

import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import matplotlib.pyplot as plt
import tensorflow as tf
from ns3gym import ns3env


# ============================================================
# 1. TEST CONFIGURATION
# ============================================================
TEST_EPISODES = 150          # Number of random seeds/test runs
MAX_STEPS = 1000
N_AGENTS = 5
STATE_DIM_PER_AGENT = 5
ACTION_DIM = 4

NUM_UES = 30
SIM_TIME = 10
PACKET_INTERVAL = 0.02778
PACKET_SIZE_BYTES = 1400

BASE_PORT = 5555
OUTPUT_DIR = "all_model_test_results"
CHECKPOINT_FILE = "evaluation_checkpoint.pkl"
CHECKPOINT_INTERVAL = 10  # Save evaluation checkpoint every 10 completed test episodes
RANDOM_SEED_FOR_TEST_SEEDS = 42
SEED_MIN = 201
SEED_MAX = 9999

# Training modules used to reconstruct each model.
MODEL_FILES = {
    "DDQN": "fast_ddqn_multiagent_c_5.py",
    "PPO": "ppo_learning_4.py",
    "Q-Learning": "tabular_q_learning_c_10.py",
}

# Checkpoint prefixes produced by the training scripts.
CHECKPOINT_PREFIXES = {
    "DDQN": "trained_ddqn_per_agent",
    "PPO": "trained_ppo_agent",
    "Q-Learning": "trained_q_agent",
}


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================
def get_total_bits():
    packets_per_ue = SIM_TIME / PACKET_INTERVAL
    total_packets = packets_per_ue * NUM_UES
    return total_packets * PACKET_SIZE_BYTES * 8


def parse_info(info):
    """Parse ns3-gym info string: total_energy=...;global_sinr=...;active_ue=...;total_power=..."""
    if isinstance(info, str):
        info_str = info
    else:
        info_str = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else ""

    info_parts = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            info_parts[key.strip()] = value.strip()
    return info_parts


def safe_float(info_parts, key, default=0.0):
    try:
        return float(info_parts.get(key, default))
    except Exception:
        return default


def safe_int(info_parts, key, default=0):
    try:
        return int(float(info_parts.get(key, default)))
    except Exception:
        return default


def split_obs_common(obs):
    return [
        np.asarray(obs[i * STATE_DIM_PER_AGENT:(i + 1) * STATE_DIM_PER_AGENT], dtype=np.float32)
        for i in range(N_AGENTS)
    ]


def dynamic_import(module_name, file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find model file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_model_path(filename):
    """
    Looks first beside this testing script, then in the current working directory.
    This makes the script easier to run from either location.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / filename, Path.cwd() / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def reward_to_float(reward):
    if isinstance(reward, (float, int, np.floating)):
        return float(reward)
    arr = np.asarray(reward, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================
def load_ddqn(ddqn_module):
    wrapper = ddqn_module.MultiAgentDDQNWrapper(
        n_agents=N_AGENTS,
        state_dim_per_agent=STATE_DIM_PER_AGENT,
        action_dim=ACTION_DIM,
    )
    wrapper.load_all(CHECKPOINT_PREFIXES["DDQN"])
    return wrapper


def load_ppo(ppo_module):
    wrapper = ppo_module.MultiAgentPPOWrapper(
        n_agents=N_AGENTS,
        state_dim_per_agent=STATE_DIM_PER_AGENT,
        action_dim=ACTION_DIM,
    )
    wrapper.load_all(CHECKPOINT_PREFIXES["PPO"])
    return wrapper


def load_qlearning(q_module):
    wrapper = q_module.MultiAgentWrapper(N_AGENTS)
    wrapper.load_all(CHECKPOINT_PREFIXES["Q-Learning"])
    return wrapper


# ============================================================
# 4. DETERMINISTIC ACTION SELECTION
# ============================================================
def ddqn_actions(wrapper, agent_states):
    # epsilon=0.0 means greedy action selection, no exploration.
    return wrapper.act(agent_states, epsilon=0.0)


def qlearning_actions(wrapper, agent_states):
    # current_epsilon=0.0 means greedy Q-table action selection, no exploration.
    return wrapper.act(agent_states, current_epsilon=0.0)


def ppo_actions_deterministic(wrapper, agent_states):
    """
    PPO's training act() samples stochastically from the policy.
    For testing, use argmax probability after applying the same transition action mask.
    """
    actions = []
    for i, agent in enumerate(wrapper.agents):
        state = np.asarray(agent_states[i], dtype=np.float32)
        is_transitioning = bool(state[-1] > 0.5)

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        probs = agent.actor(state_tensor, training=False).numpy()[0]
        probs = agent._apply_action_mask_numpy(probs, is_transitioning)

        actions.append(int(np.argmax(probs)))
    return actions


def choose_actions(model_name, wrapper, agent_states):
    if model_name == "DDQN":
        return ddqn_actions(wrapper, agent_states)
    if model_name == "PPO":
        return ppo_actions_deterministic(wrapper, agent_states)
    if model_name == "Q-Learning":
        return qlearning_actions(wrapper, agent_states)
    if model_name == "Baseline":
        return [0] * N_AGENTS  # Always ACTIVE
    raise ValueError(f"Unknown model_name: {model_name}")


# ============================================================
# 5. EVALUATION EPISODE
# ============================================================
def run_episode(model_name, wrapper, seed, port=BASE_PORT, max_steps=MAX_STEPS):
    env = ns3env.Ns3Env(
        port=port,
        stepTime=0.01,
        startSim=True,
        simSeed=int(seed),
    )

    obs = env.reset()
    agent_states = split_obs_common(obs)

    total_energy = 0.0
    total_power = 0.0
    sinr_sum = 0.0
    total_reward = 0.0
    active_ue_sum = 0.0
    step_count = 0
    done = False

    sbs_states_per_step = []
    active_ue_per_step = []

    while not done and step_count < max_steps:
        actions = choose_actions(model_name, wrapper, agent_states)
        next_obs, reward, done, info = env.step(np.asarray(actions, dtype=np.uint32))

        info_parts = parse_info(info)
        total_energy = safe_float(info_parts, "total_energy", total_energy)
        total_power = safe_float(info_parts, "total_power", total_power)
        global_sinr = safe_float(info_parts, "global_sinr", 0.0)
        active_ue = safe_int(info_parts, "active_ue", 0)

        sinr_sum += global_sinr
        total_reward += reward_to_float(reward)
        active_ue_sum += active_ue

        agent_states = split_obs_common(next_obs)
        sbs_states_per_step.append([int(state[1]) for state in agent_states])
        active_ue_per_step.append(active_ue)

        step_count += 1

    env.close()

    avg_sinr = sinr_sum / step_count if step_count > 0 else 0.0
    avg_active_ue = active_ue_sum / step_count if step_count > 0 else 0.0
    total_bits = get_total_bits()
    energy_efficiency = total_bits / total_energy if total_energy > 0 else 0.0

    return {
        "seed": int(seed),
        "energy": float(total_energy),
        "power": float(total_power),
        "sinr": float(avg_sinr),
        "reward": float(total_reward),
        "efficiency": float(energy_efficiency),
        "avg_active_ue": float(avg_active_ue),
        "steps": int(step_count),
        "active_ue_per_step": active_ue_per_step,
        "sbs_states_per_step": sbs_states_per_step,
    }


# ============================================================
# 6. CHECKPOINTING
# ============================================================
def checkpoint_path():
    return Path(OUTPUT_DIR) / CHECKPOINT_FILE


def save_checkpoint(results, seeds, config):
    """Save evaluation progress to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = checkpoint_path().with_suffix(".tmp")
    data = {
        "results": results,
        "seeds": list(seeds),
        "config": dict(config),
    }
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, checkpoint_path())


def load_checkpoint():
    path = checkpoint_path()
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def result_exists(results, model_name, seed):
    return any(int(r.get("seed", -1)) == int(seed) for r in results.get(model_name, []))


def sort_results_by_seed_order(results, seeds):
    seed_index = {int(seed): i for i, seed in enumerate(seeds)}
    for model_name in results:
        results[model_name] = sorted(
            results[model_name],
            key=lambda r: seed_index.get(int(r.get("seed", -1)), 10**9)
        )
    return results


# ============================================================
# 7. SAVE RESULTS
# ============================================================
def save_results(results, seeds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = ["energy", "power", "sinr", "reward", "efficiency", "avg_active_ue", "steps"]

    for model_name, model_results in results.items():
        safe_name = model_name.replace("-", "_").replace(" ", "_").lower()
        model_dir = Path(OUTPUT_DIR) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            arr = np.asarray([r[metric] for r in model_results], dtype=np.float64)
            np.save(model_dir / f"{metric}.npy", arr)

        # Per-model CSV summary
        with open(model_dir / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed"] + metrics)
            for r in model_results:
                writer.writerow([r["seed"]] + [r[m] for m in metrics])

    # Combined CSV summary
    combined_path = Path(OUTPUT_DIR) / "combined_summary.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed"] + metrics)
        for model_name, model_results in results.items():
            for r in model_results:
                writer.writerow([model_name, r["seed"]] + [r[m] for m in metrics])

    # Save the seed list for reproducibility
    np.save(Path(OUTPUT_DIR) / "test_seeds.npy", np.asarray(seeds, dtype=np.int32))


# ============================================================
# 8. PLOTTING
# ============================================================
def plot_metric(results, metric, ylabel, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x = np.arange(1, len(next(iter(results.values()))) + 1)

    plt.figure(figsize=(12, 6))
    for model_name, model_results in results.items():
        y = [r[metric] for r in model_results]
        plt.plot(x, y, marker="o", linewidth=2, markersize=4, label=model_name)

    plt.xlabel("Test seed index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / filename)
    plt.close()


def plot_combined_dashboard(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x = np.arange(1, len(next(iter(results.values()))) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_specs = [
        ("energy", "Energy (J)", "Total Energy"),
        ("sinr", "SINR (dB)", "Average SINR"),
        ("efficiency", "Bits/J", "Energy Efficiency"),
        ("reward", "Cumulative Reward", "Total Reward"),
    ]

    for ax, (metric, ylabel, title) in zip(axes.flat, plot_specs):
        for model_name, model_results in results.items():
            y = [r[metric] for r in model_results]
            ax.plot(x, y, marker="o", linewidth=2, markersize=3, label=model_name)
        ax.set_xlabel("Test seed index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    fig.suptitle("DDQN vs PPO vs Q-Learning vs Baseline", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(Path(OUTPUT_DIR) / "combined_all_metrics_comparison.png")
    plt.close(fig)


def plot_average_bar_summary(results):
    """One compact average-performance graph for report-style comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = ["energy", "sinr", "efficiency", "reward"]
    labels = ["Energy", "SINR", "Efficiency", "Reward"]
    model_names = list(results.keys())

    summary = {metric: [np.mean([r[metric] for r in results[m]]) for m in model_names] for metric in metrics}

    # Normalize each metric to 0-1 for a single combined graph.
    # For energy, lower is better, so invert it after normalization.
    normalized = {}
    for metric in metrics:
        values = np.asarray(summary[metric], dtype=np.float64)
        vmin, vmax = values.min(), values.max()
        if abs(vmax - vmin) < 1e-12:
            norm = np.ones_like(values)
        else:
            norm = (values - vmin) / (vmax - vmin)
        if metric == "energy":
            norm = 1.0 - norm
        normalized[metric] = norm

    x = np.arange(len(model_names))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, normalized[metric], width, label=labels[i])

    plt.xticks(x, model_names)
    plt.ylabel("Normalized score (higher is better)")
    plt.title("Average Normalized Performance Summary")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "average_normalized_summary.png")
    plt.close()


def generate_plots(results):
    plot_metric(results, "energy", "Energy (J)", "Total Energy per Test Seed", "compare_energy.png")
    plot_metric(results, "power", "Power (W)", "Total Power per Test Seed", "compare_power.png")
    plot_metric(results, "sinr", "SINR (dB)", "Average SINR per Test Seed", "compare_sinr.png")
    plot_metric(results, "efficiency", "Bits/J", "Energy Efficiency per Test Seed", "compare_efficiency.png")
    plot_metric(results, "reward", "Cumulative Reward", "Total Reward per Test Seed", "compare_reward.png")
    plot_metric(results, "avg_active_ue", "Average active UEs", "Average Active UEs per Test Seed", "compare_active_ue.png")
    plot_combined_dashboard(results)
    plot_average_bar_summary(results)


# ============================================================
# 9. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES, help="Number of test seeds/episodes.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum steps per episode.")
    parser.add_argument("--port", type=int, default=BASE_PORT, help="ns3-gym port.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_FOR_TEST_SEEDS, help="Seed used to generate test seeds.")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL, help="Save checkpoint every N completed test episodes. Default: 10.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from evaluation checkpoint if it exists. Default: enabled.")
    parser.add_argument("--restart", action="store_true", help="Ignore/delete any existing evaluation checkpoint and start from scratch.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.restart and checkpoint_path().exists():
        checkpoint_path().unlink()
        print(f">>> Removed old checkpoint: {checkpoint_path()}")

    random.seed(args.seed)
    seeds = random.sample(range(SEED_MIN, SEED_MAX), args.episodes)

    config = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "port": args.port,
        "seed": args.seed,
        "n_agents": N_AGENTS,
        "state_dim_per_agent": STATE_DIM_PER_AGENT,
        "action_dim": ACTION_DIM,
        "checkpoint_interval": args.checkpoint_interval,
    }

    checkpoint = load_checkpoint() if args.resume and not args.restart else None

    print("=== Loading model modules ===")
    ddqn_module = dynamic_import("ddqn_latest", resolve_model_path(MODEL_FILES["DDQN"]))
    ppo_module = dynamic_import("ppo_latest", resolve_model_path(MODEL_FILES["PPO"]))
    q_module = dynamic_import("qlearning_latest", resolve_model_path(MODEL_FILES["Q-Learning"]))

    print("=== Loading trained checkpoints ===")
    models = {
        "DDQN": load_ddqn(ddqn_module),
        "PPO": load_ppo(ppo_module),
        "Q-Learning": load_qlearning(q_module),
        "Baseline": None,
    }

    if checkpoint is not None:
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_seeds = checkpoint.get("seeds", [])
        if checkpoint_config == config and list(checkpoint_seeds) == list(seeds):
            results = checkpoint.get("results", {model_name: [] for model_name in models})
            for model_name in models:
                results.setdefault(model_name, [])
            print(f">>> Resuming evaluation from checkpoint: {checkpoint_path()}")
        else:
            print(">>> Existing checkpoint does not match this run configuration. Starting fresh.")
            results = {model_name: [] for model_name in models}
    else:
        results = {model_name: [] for model_name in models}

    def completed_count():
        return sum(len(model_results) for model_results in results.values())

    completed_since_checkpoint = 0

    print("=== Starting evaluation ===")
    try:
        for model_name, wrapper in models.items():
            print(f"\n--- Evaluating {model_name} ---")
            for idx, seed in enumerate(seeds, start=1):
                if result_exists(results, model_name, seed):
                    print(f"[{model_name}] Test {idx}/{len(seeds)} | simSeed={seed} already completed, skipping.")
                    continue

                print(f"[{model_name}] Test {idx}/{len(seeds)} | simSeed={seed}")
                episode_result = run_episode(
                    model_name=model_name,
                    wrapper=wrapper,
                    seed=seed,
                    port=args.port,
                    max_steps=args.max_steps,
                )
                results[model_name].append(episode_result)
                results = sort_results_by_seed_order(results, seeds)
                completed_since_checkpoint += 1

                if completed_since_checkpoint >= args.checkpoint_interval:
                    save_checkpoint(results, seeds, config)
                    print(
                        f"    Saved checkpoint after {completed_count()} completed test episodes "
                        f"({model_name} seed {seed})."
                    )
                    completed_since_checkpoint = 0

                # Small pause helps avoid port/process cleanup issues in some ns3-gym setups.
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n>>> Evaluation interrupted by user. Saving latest completed results before exit...")
        results = sort_results_by_seed_order(results, seeds)
        save_checkpoint(results, seeds, config)
        save_results(results, seeds)
        print(f">>> Saved interrupt checkpoint to: {checkpoint_path()}")
        print(f">>> Partial results saved in: {OUTPUT_DIR}")
        raise

    except Exception:
        print("\n>>> Evaluation stopped because of an error. Saving latest completed results before exit...")
        results = sort_results_by_seed_order(results, seeds)
        save_checkpoint(results, seeds, config)
        save_results(results, seeds)
        print(f">>> Saved error checkpoint to: {checkpoint_path()}")
        print(f">>> Partial results saved in: {OUTPUT_DIR}")
        raise

    results = sort_results_by_seed_order(results, seeds)

    # Final checkpoint, even if the last batch is fewer than checkpoint_interval.
    save_checkpoint(results, seeds, config)
    print(f"\n>>> Final checkpoint saved after {completed_count()} completed test episodes.")

    print("\n=== Saving results ===")
    save_results(results, seeds)

    print("=== Generating plots ===")
    generate_plots(results)

    print(f"\nDone. Results saved in: {OUTPUT_DIR}")
    print("Main comparison graph:", Path(OUTPUT_DIR) / "combined_all_metrics_comparison.png")
    print("Summary CSV:", Path(OUTPUT_DIR) / "combined_summary.csv")
    print("Checkpoint file kept at:", checkpoint_path())


if __name__ == "__main__":
    main()
