import os
import csv
import random
from collections import deque

import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import matplotlib.pyplot as plt
from ns3gym import ns3env


# ============================================================
# 1. GLOBAL PATH CONFIGURATION
# ============================================================
OUTPUT_DIR = "qlearning_results_c_10_500_2"
CHECKPOINT_PREFIX = "trained_q_agent"


# ============================================================
# 2. HYPERPARAMETERS & ENVIRONMENT PARAMETERS
# ============================================================
EPISODES = 500
MAX_STEPS = 1000

N_AGENTS = 5
STATE_DIM_PER_AGENT = 5
ACTION_DIM = 4

NUM_UES = 30
SIM_TIME = 10
PACKET_INTERVAL = 0.02778
PACKET_SIZE_BYTES = 1400

ALPHA = 0.01
GAMMA = 0.90
EPSILON = 1.0
EPS_DECAY = 0.9995
EPS_MIN = 0.005

RESUME = True


# ============================================================
# 3. DISCRETIZATION BINS
#    Observation structure from your environment:
#    [active_ues, state, power, global_sinr, transitioning]
# ============================================================
UE_BINS = np.array([0, 3, 6, 10, 15, 25], dtype=np.float32)
STATE_BINS = np.array([0.5, 1.5, 2.5], dtype=np.float32)   # 0,1,2,3
POWER_BINS = np.array([0, 3, 7, 12, 18], dtype=np.float32)
SINR_BINS = np.array([-5, 0, 5, 13, 20, 25], dtype=np.float32)

NUM_UE_LEVELS = len(UE_BINS) + 1
NUM_STATE_LEVELS = len(STATE_BINS) + 1
NUM_POWER_LEVELS = len(POWER_BINS) + 1
NUM_SINR_LEVELS = len(SINR_BINS) + 1
NUM_TRANSITION_LEVELS = 2

STATE_SPACE_SIZE = (
    NUM_UE_LEVELS
    * NUM_STATE_LEVELS
    * NUM_POWER_LEVELS
    * NUM_SINR_LEVELS
    * NUM_TRANSITION_LEVELS
)


# ============================================================
# 4. SMALL HELPERS
# ============================================================
def split_obs(obs, n_agents=N_AGENTS, state_dim_per_agent=STATE_DIM_PER_AGENT):
    return [
        np.asarray(obs[i * state_dim_per_agent:(i + 1) * state_dim_per_agent], dtype=np.float32)
        for i in range(n_agents)
    ]


def parse_info(info):
    if isinstance(info, str):
        info_str = info
    else:
        info_str = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else ""

    info_parts = {}
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            info_parts[k.strip()] = v.strip()
    return info_parts


def safe_float(info_parts, key, default=0.0):
    try:
        return float(info_parts.get(key, default))
    except Exception:
        return default


def get_total_bits():
    packets_per_ue = SIM_TIME / PACKET_INTERVAL
    total_packets = packets_per_ue * NUM_UES
    return total_packets * PACKET_SIZE_BYTES * 8


def save_ddqn_style_final_episode_graphs(
    output_dir,
    algorithm_label,
    avg_rewards_per_episode,
    total_energy_per_episode,
    avg_sbs_sinr_per_episode,
):
    """Save final per-episode arrays and plots."""
    os.makedirs(output_dir, exist_ok=True)
    total_bits = get_total_bits()
    ee_per_episode = [total_bits / e if e > 0 else 0.0 for e in total_energy_per_episode]

    np.save(os.path.join(output_dir, "avg_reward_per_episode.npy"), np.array(avg_rewards_per_episode))
    np.save(os.path.join(output_dir, "sinr_per_episode.npy"), np.array(avg_sbs_sinr_per_episode))
    np.save(os.path.join(output_dir, "energy_efficiency_per_episode.npy"), np.array(ee_per_episode))
    np.save(os.path.join(output_dir, "rl_energy_per_episode.npy"), np.array(total_energy_per_episode))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ee_per_episode) + 1), ee_per_episode, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bits/J)")
    plt.title("Energy Efficiency per Episode (Q-Learning)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_efficiency_per_episode.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_sbs_sinr_per_episode) + 1), avg_sbs_sinr_per_episode, label="Global Average SINR")
    plt.xlabel("Episode")
    plt.ylabel("Global SINR (dB)")
    plt.title("Global SINR per Episode (Q-Learning)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sinr_per_episode.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_per_episode, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward per Episode ({algorithm_label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_reward_per_episode.png"))
    plt.close()


# ============================================================
# 5. TABULAR Q-LEARNING AGENT
# ============================================================
class TabularQLearningAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.q_table = np.full((STATE_SPACE_SIZE, ACTION_DIM), -1.0, dtype=np.float32)
        self.loss_history = []

    def discretize(self, obs):
        """
        Converts one SBS observation vector into a compact discrete state index.
        obs = [active_ues, state, power, global_sinr, transitioning]
        """
        active_ues = float(obs[0])
        sbs_state = float(obs[1])
        power = float(obs[2])
        sinr = float(obs[3])
        transitioning = float(obs[4])

        ue_idx = np.digitize(active_ues, UE_BINS)
        state_idx = np.digitize(sbs_state, STATE_BINS)
        power_idx = np.digitize(power, POWER_BINS)
        sinr_idx = np.digitize(sinr, SINR_BINS)
        trans_idx = 1 if transitioning > 0.5 else 0

        idx = ue_idx
        idx = idx * NUM_STATE_LEVELS + state_idx
        idx = idx * NUM_POWER_LEVELS + power_idx
        idx = idx * NUM_SINR_LEVELS + sinr_idx
        idx = idx * NUM_TRANSITION_LEVELS + trans_idx
        return int(idx)

    def valid_actions(self, obs):
        """
        Keep action 0 if the SBS is transitioning.
        Otherwise all actions are valid.
        """
        if float(obs[4]) > 0.5:
            return [0]
        return list(range(ACTION_DIM))

    def act(self, state_idx, obs, current_epsilon):
        valid = self.valid_actions(obs)

        if np.random.rand() < current_epsilon:
        # Prefer ACTIVE during exploration
            if 0 in valid and np.random.rand() < 0.7:
                return 0
            return random.choice(valid)

        q_values = self.q_table[state_idx].copy()
        masked_q = np.full_like(q_values, -np.inf)
        for a in valid:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def update(self, s, a, r, s_next, done, next_obs):
        if done:
            best_next_q = 0.0
        else:
            valid_next = self.valid_actions(next_obs)
            best_next_q = np.max(self.q_table[s_next][valid_next])

        td_target = r + GAMMA * best_next_q
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += ALPHA * td_error
        return float(td_error)

    def save(self, path_prefix, episode, current_epsilon):
        np.save(f"{path_prefix}_qtable.npy", self.q_table)
        meta = {
            "epsilon": current_epsilon,
            "last_episode": episode,
        }
        np.save(f"{path_prefix}_meta.npy", meta)

    def load(self, path_prefix):
        self.q_table = np.load(f"{path_prefix}_qtable.npy")
        meta = np.load(f"{path_prefix}_meta.npy", allow_pickle=True).item()
        return meta.get("epsilon", 1.0), meta.get("last_episode", 0)


# ============================================================
# 6. MULTI-AGENT WRAPPER
# ============================================================
class MultiAgentWrapper:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agents = [TabularQLearningAgent(i) for i in range(n_agents)]

    def split_obs(self, obs):
        return split_obs(obs, self.n_agents, STATE_DIM_PER_AGENT)

    def act(self, agent_states, current_epsilon):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(
                state_idx=agent.discretize(agent_states[i]),
                obs=agent_states[i],
                current_epsilon=current_epsilon,
            )
            actions.append(action)
        return actions

    def update(self, agent_states, actions, reward, next_agent_states, done):
        td_errors = []
        for i, agent in enumerate(self.agents):
            s = agent.discretize(agent_states[i])
            s_next = agent.discretize(next_agent_states[i])
            td_error = agent.update(s, actions[i], reward, s_next, done, next_agent_states[i])
            td_errors.append(td_error)
        return td_errors

    def save_all(self, base_path="agent", episode=0, current_epsilon=1.0, histories=None):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            agent.save(full_path, episode, current_epsilon)

        if histories:
            for name, data in histories.items():
                save_path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
                np.save(save_path, np.array(data))

    def load_all(self, base_path="agent"):
        last_ep = 0
        current_eps = 1.0
        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            current_eps, last_ep = agent.load(full_path)

        hist_data = {}
        for name in ["avg_rewards", "energy", "sinr", "power", "td_error"]:
            path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
            if os.path.exists(path):
                hist_data[name] = list(np.load(path))
        return last_ep, current_eps, hist_data

    def save_all_qtables(self, base_path="agent"):
        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            np.save(f"{full_path}_qtable.npy", agent.q_table)


# ============================================================
# 7. BASELINE SIMULATION
# ============================================================
def simulate_baseline_energy_and_sinr(env, episodes, max_steps):
    baseline_energy = []
    baseline_sinr = []
    baseline_power = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        total_energy = 0.0
        total_power = 0.0
        sinr_sum = 0.0
        sinr_steps = 0

        while not done and step_count < max_steps:
            action = [0] * N_AGENTS
            _, _, done, info = env.step(np.array(action, dtype=np.uint32))

            info_parts = parse_info(info)
            total_energy = safe_float(info_parts, "total_energy", 0.0)
            total_power = safe_float(info_parts, "total_power", 0.0)
            global_sinr = safe_float(info_parts, "global_sinr", 0.0)

            sinr_sum += global_sinr
            sinr_steps += 1
            step_count += 1

        baseline_energy.append(total_energy)
        baseline_power.append(total_power)
        baseline_sinr.append(sinr_sum / sinr_steps if sinr_steps else 0.0)

    return baseline_energy, baseline_sinr, baseline_power


# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wrapper = MultiAgentWrapper(N_AGENTS)

    avg_rewards_per_episode = []
    total_energy_per_episode = []
    total_power_per_episode = []
    avg_sbs_sinr_per_episode = []
    td_error_history = []
    rewards_history = []
    step_rewards = []
    sbs_state_log = {i: [] for i in range(N_AGENTS)}

    start_episode = 1
    ep = 0

    expected_meta = os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_0_meta.npy")
    if RESUME and os.path.exists(expected_meta):
        print(f">>> SUCCESS: Checkpoint found in {OUTPUT_DIR}. Resuming Q-Learning...")
        last_saved_episode, current_epsilon, histories = wrapper.load_all(CHECKPOINT_PREFIX)
        start_episode = last_saved_episode + 1
        ep = last_saved_episode

        avg_rewards_per_episode = histories.get("avg_rewards", [])
        total_energy_per_episode = histories.get("energy", [])
        avg_sbs_sinr_per_episode = histories.get("sinr", [])
        total_power_per_episode = histories.get("power", [])
        td_error_history = histories.get("td_error", [])

        print(f">>> Resuming from Episode {start_episode}. Epsilon: {current_epsilon:.4f}")
    else:
        current_epsilon = EPSILON
        print(f">>> NOTICE: No checkpoint in {OUTPUT_DIR}. Starting fresh.")

    print("==== Tabular Q-Learning Training Start ====")

    env = ns3env.Ns3Env(port=5557, stepTime=0.01, startSim=True, simSeed=1)

    try:
        for ep in range(start_episode, EPISODES + 1):
            env.simSeed = ep
            obs = env.reset()
            agent_states = wrapper.split_obs(obs)

            done = False
            step_count = 0
            episode_reward = np.zeros(N_AGENTS, dtype=np.float32)

            current_episode_energy = 0.0
            current_episode_power = 0.0
            sbs_sinr_sum = 0.0
            sinr_step_count = 0

            while not done and step_count < MAX_STEPS:
                print("--------------------")
                print(f"Step {step_count + 1} (Episode {ep})")
                print("--------------------")

                actions = wrapper.act(agent_states, current_epsilon)
                next_obs, reward, done, info = env.step(np.array(actions, dtype=np.uint32))
                next_agent_states = wrapper.split_obs(next_obs)

                for i in range(N_AGENTS):
                    current_state = int(next_obs[i * STATE_DIM_PER_AGENT + 1])
                    sbs_state_log[i].append((ep, step_count, current_state))

                info_parts = parse_info(info)
                print(f"Step {step_count + 1} | Reward: {reward} | Done: {done} | "
                      f"Info: {info_parts} | Next Obs: {next_obs}")
                print(f"Step result: next_obs={next_obs}, reward={reward}, done={done}, info={info}")

                current_episode_energy = safe_float(info_parts, "total_energy", current_episode_energy)
                current_episode_power = safe_float(info_parts, "total_power", current_episode_power)
                global_sinr_value = safe_float(info_parts, "global_sinr", 0.0)

                sbs_sinr_sum += global_sinr_value
                sinr_step_count += 1

                reward_value = float(reward) if isinstance(reward, (float, int, np.floating)) else float(np.mean(reward))
                rewards = [reward_value] * N_AGENTS

                td_errors = wrapper.update(agent_states, actions, reward_value, next_agent_states, done)
                td_error_history.append(float(np.mean(td_errors)))

                agent_states = next_agent_states
                episode_reward += np.array(rewards, dtype=np.float32)
                step_count += 1
                current_epsilon = max(EPS_MIN, current_epsilon * EPS_DECAY)

            avg_reward = float(np.mean(episode_reward))
            rewards_history.append(episode_reward.tolist())
            avg_rewards_per_episode.append(avg_reward)
            total_energy_per_episode.append(current_episode_energy)
            total_power_per_episode.append(current_episode_power)
            avg_sbs_sinr_per_episode.append(sbs_sinr_sum / sinr_step_count if sinr_step_count > 0 else 0.0)

            

            print(f"Episode {ep}: Reward: {episode_reward} | Epsilon: {current_epsilon:.4f}")

            total_bits = get_total_bits()
            ee_per_episode = [
                total_bits / e if e > 0 else 0.0
                for e in total_energy_per_episode
            ]

            plt.figure()
            plt.plot(avg_rewards_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Average Reward per Episode (Q-Learning)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "avg_reward_per_episode_live.png"))
            plt.close()

            plt.figure()
            plt.plot(avg_sbs_sinr_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Global SINR (dB)")
            plt.title("Global SINR per Episode (Q-Learning)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "sinr_per_episode_live.png"))
            plt.close()

            plt.figure()
            plt.plot(ee_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Energy Efficiency (bits/J)")
            plt.title("Energy Efficiency per Episode (Q-Learning)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode_live.png"))
            plt.close()

            if ep % 10 == 0:
                hist_to_save = {
                    "avg_rewards": avg_rewards_per_episode,
                    "energy": total_energy_per_episode,
                    "sinr": avg_sbs_sinr_per_episode,
                    "power": total_power_per_episode,
                    "td_error": td_error_history,
                }
                wrapper.save_all(CHECKPOINT_PREFIX, ep, current_epsilon, histories=hist_to_save)
                print(f"Checkpoint saved at Episode {ep}")

        env.close()

        wrapper.save_all(CHECKPOINT_PREFIX, ep, current_epsilon, histories={
            "avg_rewards": avg_rewards_per_episode,
            "energy": total_energy_per_episode,
            "sinr": avg_sbs_sinr_per_episode,
            "power": total_power_per_episode,
            "td_error": td_error_history,
        })
        print("Final model saved.")

        csv_path = os.path.join(OUTPUT_DIR, "sbs_state_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step"] + [f"SBS_{i}_state" for i in range(N_AGENTS)])

            max_len = max(len(sbs_state_log[i]) for i in sbs_state_log)
            for idx in range(max_len):
                try:
                    ep_val, step = sbs_state_log[0][idx][:2]
                    row = [ep_val, step] + [sbs_state_log[i][idx][2] for i in range(N_AGENTS)]
                    writer.writerow(row)
                except IndexError:
                    continue

        print(f"State history saved to {csv_path}")

        os.environ["NS3_BASELINE"] = "1"
        print("\n[INFO] Launching baseline simulation for each episode...")
        baseline_energy_per_episode = []
        baseline_power_per_episode = []
        baseline_sinr_per_episode = []

        for b_ep in range(1, EPISODES + 1):
            print(f"[Baseline] Running Episode {b_ep} with simSeed={b_ep}")
            env = ns3env.Ns3Env(port=5557, stepTime=0.01, startSim=True, simSeed=b_ep)
            energy, sinr, power = simulate_baseline_energy_and_sinr(env, 1, MAX_STEPS)
            baseline_energy_per_episode.extend(energy)
            baseline_power_per_episode.extend(power)
            baseline_sinr_per_episode.extend(sinr)
            env.close()

        os.environ["NS3_BASELINE"] = "0"

        total_bits = get_total_bits()

        np.save(os.path.join(OUTPUT_DIR, "rl_energy_qlearning.npy"), np.array(total_energy_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "baseline_energy_qlearning.npy"), np.array(baseline_energy_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "baseline_power_qlearning.npy"), np.array(baseline_power_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "baseline_sinr_qlearning.npy"), np.array(baseline_sinr_per_episode))

        # Save generic filenames used by the analysis scripts.
        np.save(os.path.join(OUTPUT_DIR, "baseline_energy_per_episode.npy"), np.array(baseline_energy_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "baseline_power_per_episode.npy"), np.array(baseline_power_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "baseline_sinr_per_episode.npy"), np.array(baseline_sinr_per_episode))
        np.save(os.path.join(OUTPUT_DIR, "energy_efficiency_baseline.npy"), np.array([
            total_bits / e if e > 0 else 0.0 for e in baseline_energy_per_episode
        ]))

        metrics = [
            (total_energy_per_episode, baseline_energy_per_episode, "Energy Consumption Comparison (Q-Learning)", "Total Energy (J)", "energy_comparison.png"),
            (total_power_per_episode, baseline_power_per_episode, "Power Comparison (Q-Learning)", "Total Power (W)", "power_comparison.png"),
            (avg_sbs_sinr_per_episode, baseline_sinr_per_episode, "SINR Comparison (Q-Learning)", "Global SINR (dB)", "sinr_comparison.png"),
            ([total_bits / e if e > 0 else 0.0 for e in total_energy_per_episode],
             [total_bits / e if e > 0 else 0.0 for e in baseline_energy_per_episode],
             "Energy Efficiency Comparison (Q-Learning)", "Energy Efficiency (bits/J)", "energy_efficiency_comparison.png")
        ]

        for rl, base, title, ylabel, fname in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(rl, label="RL")
            plt.plot(base, label="Baseline", linestyle="--")
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

        save_ddqn_style_final_episode_graphs(
            output_dir=OUTPUT_DIR,
            algorithm_label="Q-Learning",
            avg_rewards_per_episode=avg_rewards_per_episode,
            total_energy_per_episode=total_energy_per_episode,
            avg_sbs_sinr_per_episode=avg_sbs_sinr_per_episode,
        )

        print("All DDQN-style plots and checkpoints saved successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        wrapper.save_all(CHECKPOINT_PREFIX, ep, current_epsilon, histories={
            "avg_rewards": avg_rewards_per_episode,
            "energy": total_energy_per_episode,
            "sinr": avg_sbs_sinr_per_episode,
            "power": total_power_per_episode,
            "td_error": td_error_history,
        })
        env.close()
        print("Saved and exited cleanly.")
