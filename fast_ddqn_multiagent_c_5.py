import os
import csv
import random
from collections import deque

import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from ns3gym import ns3env


# ============================================================
# 1. GLOBAL PATH CONFIGURATION
# ============================================================
OUTPUT_DIR = "ddqn_per_results_c_5_500_2"
CHECKPOINT_PREFIX = "trained_ddqn_per_agent"


# ============================================================
# 2. HYPERPARAMETERS & ENVIRONMENT PARAMETERS
# ============================================================
EPISODES = 500
MAX_STEPS = 1000

N_AGENTS = 5
STATE_DIM_PER_AGENT = 5

# Actions 0-3 correspond to ACTIVE, SM1, SM2, and SM3.
ACTION_DIM = 4

NUM_UES = 30
SIM_TIME = 10
PACKET_INTERVAL = 0.02778
PACKET_SIZE_BYTES = 1400

LEARNING_RATE = 0.0003
GAMMA = 0.98
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 100
MEMORY_CAPACITY = 100000

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

PER_ALPHA = 0.30
PER_BETA_START = 0.70
PER_BETA_FRAMES = 50000
PER_EPS = 1e-6

RESUME = True


# ============================================================
# 3. SMALL HELPERS
# ============================================================
def split_obs(obs, n_agents=N_AGENTS, state_dim_per_agent=STATE_DIM_PER_AGENT):
    return [
        np.asarray(obs[i * state_dim_per_agent:(i + 1) * state_dim_per_agent], dtype=np.float32)
        for i in range(n_agents)
    ]


def parse_info(info):
    """
    Robust parsing of the OpenGym extra info string:
    total_energy=...;global_sinr=...;active_ue=...;total_power=...
    """
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


# ============================================================
# 4. PRIORITIZED EXPERIENCE REPLAY BUFFER
# ============================================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0

        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]

        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = np.asarray(weights, dtype=np.float32)

        states = np.asarray([s for s, _, _, _, _ in samples], dtype=np.float32)
        actions = np.asarray([a for _, a, _, _, _ in samples], dtype=np.int32)
        rewards = np.asarray([r for _, _, r, _, _ in samples], dtype=np.float32)
        next_states = np.asarray([ns for _, _, _, ns, _ in samples], dtype=np.float32)
        dones = np.asarray([d for _, _, _, _, d in samples], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.asarray(td_errors, dtype=np.float32)
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + PER_EPS


# ============================================================
# 5. DDQN AGENT WITH PER
# ============================================================
class DDQNPERAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id

        self.replay_memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, PER_ALPHA)

        self.online_q_network = self._build_network()
        self.target_q_network = self._build_network()
        self.update_target_network()

        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        self.train_steps = 0
        self.loss_history = []
        self.td_error_history = []

    def _build_network(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        return model

    def update_target_network(self):
        self.target_q_network.set_weights(self.online_q_network.get_weights())

    def valid_actions(self, state):
        """
        If your state vector uses the final value as a transition flag:
        state = [active_ues, sbs_state, power, global_sinr, transitioning]

        During transition, force ACTIVE action 0.
        Otherwise allow all actions.
        """
        is_transitioning = bool(float(state[-1]) > 0.5)

        if is_transitioning:
            return [0]

        return list(range(self.action_size))

    def act(self, state, epsilon):
        valid = self.valid_actions(state)

        if np.random.rand() < epsilon:
            return random.choice(valid)

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q_values = self.online_q_network(state_tensor, training=False).numpy()[0]

        masked_q = np.full_like(q_values, -np.inf)
        for a in valid:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.add(state, action, reward, next_state, done)

    def beta_by_frame(self):
        beta = PER_BETA_START + self.train_steps * (1.0 - PER_BETA_START) / PER_BETA_FRAMES
        return min(1.0, beta)

    def replay(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return None

        beta = self.beta_by_frame()

        states, actions, rewards, next_states, dones, indices, weights = self.replay_memory.sample(
            BATCH_SIZE,
            beta
        )

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        # ----------------------------------------------------
        # DDQN target:
        # a_best = argmax_a Q_online(s_next, a)
        # y = r + gamma * Q_target(s_next, a_best)
        # ----------------------------------------------------
        next_q_online = self.online_q_network(next_states_tf, training=False).numpy()
        next_q_target = self.target_q_network(next_states_tf, training=False).numpy()

        best_next_actions = []
        for i in range(BATCH_SIZE):
            valid_next = self.valid_actions(next_states[i])

            masked_next_q = np.full(self.action_size, -np.inf, dtype=np.float32)
            for a in valid_next:
                masked_next_q[a] = next_q_online[i, a]

            best_next_actions.append(int(np.argmax(masked_next_q)))

        best_next_actions = np.asarray(best_next_actions, dtype=np.int32)
        target_next_q = next_q_target[np.arange(BATCH_SIZE), best_next_actions]

        targets = rewards + GAMMA * target_next_q * (1.0 - dones)
        targets_tf = tf.convert_to_tensor(targets, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.online_q_network(states_tf, training=True)
            action_one_hot = tf.one_hot(actions_tf, depth=self.action_size)
            predicted_q = tf.reduce_sum(q_values * action_one_hot, axis=1)

            td_errors = targets_tf - predicted_q
            loss = tf.reduce_mean(weights_tf * tf.square(td_errors))

        gradients = tape.gradient(loss, self.online_q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_q_network.trainable_variables))

        td_errors_np = td_errors.numpy()
        self.replay_memory.update_priorities(indices, td_errors_np)

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()

        loss_value = float(loss.numpy())
        self.loss_history.append(loss_value)
        self.td_error_history.append(float(np.mean(np.abs(td_errors_np))))

        return loss_value

    def save(self, path_prefix, episode, epsilon):
        self.online_q_network.save(f"{path_prefix}_online.keras")
        self.target_q_network.save(f"{path_prefix}_target.keras")

        meta = {
            "epsilon": epsilon,
            "train_steps": self.train_steps,
            "last_episode": episode,
        }
        np.save(f"{path_prefix}_meta.npy", meta)

    def load(self, path_prefix):
        self.online_q_network = keras.models.load_model(f"{path_prefix}_online.keras")
        self.target_q_network = keras.models.load_model(f"{path_prefix}_target.keras")

        meta = np.load(f"{path_prefix}_meta.npy", allow_pickle=True).item()
        self.train_steps = meta.get("train_steps", 0)

        return meta.get("epsilon", EPSILON_START), meta.get("last_episode", 0)

    def save_loss_history(self, path_prefix):
        np.save(f"{path_prefix}_loss.npy", np.array(self.loss_history))
        np.save(f"{path_prefix}_td_error.npy", np.array(self.td_error_history))


# ============================================================
# 6. MULTI-AGENT DDQN WRAPPER
# ============================================================
class MultiAgentDDQNWrapper:
    def __init__(self, n_agents, state_dim_per_agent, action_dim):
        self.n_agents = n_agents
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim = action_dim

        self.agents = [
            DDQNPERAgent(state_dim_per_agent, action_dim, i)
            for i in range(n_agents)
        ]

    def split_obs(self, obs):
        return split_obs(obs, self.n_agents, self.state_dim_per_agent)

    def act(self, agent_states, epsilon):
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(agent_states[i], epsilon))
        return actions

    def remember(self, agent_states, actions, rewards, next_agent_states, done):
        for i, agent in enumerate(self.agents):
            agent.remember(
                state=agent_states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_agent_states[i],
                done=done
            )

    def replay(self):
        losses = []
        for agent in self.agents:
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)

        if len(losses) == 0:
            return None

        return float(np.mean(losses))

    def save_all(self, base_path="agent", episode=0, epsilon=1.0, histories=None):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            agent.save(full_path, episode, epsilon)

        if histories:
            for name, data in histories.items():
                save_path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
                np.save(save_path, np.array(data))

    def load_all(self, base_path="agent"):
        last_ep = 0
        epsilon = EPSILON_START

        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            epsilon, last_ep = agent.load(full_path)

        hist_data = {}
        for name in ["avg_rewards", "energy", "sinr", "power", "loss"]:
            path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
            if os.path.exists(path):
                hist_data[name] = list(np.load(path))

        return last_ep, epsilon, hist_data

    def save_all_losses(self, base_path="agent"):
        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            agent.save_loss_history(full_path)


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

            total_energy = safe_float(info_parts, "total_energy", total_energy)
            total_power = safe_float(info_parts, "total_power", total_power)
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

    wrapper = MultiAgentDDQNWrapper(
        n_agents=N_AGENTS,
        state_dim_per_agent=STATE_DIM_PER_AGENT,
        action_dim=ACTION_DIM
    )

    avg_rewards_per_episode = []
    total_energy_per_episode = []
    total_power_per_episode = []
    avg_sbs_sinr_per_episode = []
    loss_history = []
    rewards_history = []
    step_rewards = []
    sbs_state_log = {i: [] for i in range(N_AGENTS)}

    start_episode = 1
    ep = 0
    epsilon = EPSILON_START

    expected_meta = os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_0_meta.npy")
    if RESUME and os.path.exists(expected_meta):
        print(f">>> SUCCESS: Checkpoint found in {OUTPUT_DIR}. Resuming DDQN...")
        last_saved_episode, epsilon, histories = wrapper.load_all(CHECKPOINT_PREFIX)
        start_episode = last_saved_episode + 1
        ep = last_saved_episode

        avg_rewards_per_episode = histories.get("avg_rewards", [])
        total_energy_per_episode = histories.get("energy", [])
        avg_sbs_sinr_per_episode = histories.get("sinr", [])
        total_power_per_episode = histories.get("power", [])
        loss_history = histories.get("loss", [])

        print(f">>> Resuming from Episode {start_episode}. Epsilon: {epsilon:.4f}")
    else:
        print(f">>> NOTICE: No checkpoint in {OUTPUT_DIR}. Starting fresh.")

    print("==== DDQN with Prioritized Experience Replay Training Start ====")

    env = ns3env.Ns3Env(port=5555, stepTime=0.01, startSim=True, simSeed=1)

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

                # 1. Action selection: epsilon-greedy
                actions = wrapper.act(agent_states, epsilon)

                # 2. Interaction with ns-3 environment
                next_obs, reward, done, info = env.step(np.array(actions, dtype=np.uint32))
                next_agent_states = wrapper.split_obs(next_obs)

                # 3. Log SBS states
                for i in range(N_AGENTS):
                    current_state = int(next_obs[i * STATE_DIM_PER_AGENT + 1])
                    sbs_state_log[i].append((ep, step_count, current_state))

                # 4. Parse info
                info_parts = parse_info(info)
                print(f"Step {step_count + 1} | Reward: {reward} | Done: {done} | "
                      f"Info: {info_parts} | Actions: {actions} | Next Obs: {next_obs}")

                current_episode_energy = safe_float(info_parts, "total_energy", current_episode_energy)
                current_episode_power = safe_float(info_parts, "total_power", current_episode_power)
                global_sinr_value = safe_float(info_parts, "global_sinr", 0.0)

                sbs_sinr_sum += global_sinr_value
                sinr_step_count += 1

                # 5. Reward handling
                if isinstance(reward, (float, int, np.floating)):
                    rewards = [float(reward)] * N_AGENTS
                else:
                    reward_arr = np.asarray(reward, dtype=np.float32)
                    if reward_arr.size == N_AGENTS:
                        rewards = reward_arr.tolist()
                    else:
                        rewards = [float(np.mean(reward_arr))] * N_AGENTS

                total_step_reward = float(np.sum(rewards))
                step_rewards.append(total_step_reward)

                # 6. Store transition into PER memory
                wrapper.remember(agent_states, actions, rewards, next_agent_states, done)

                # 7. Training step
                loss = wrapper.replay()
                if loss is not None:
                    loss_history.append(loss)

                # 8. Move to next state
                agent_states = next_agent_states
                episode_reward += np.asarray(rewards, dtype=np.float32)
                step_count += 1

                # 9. Decay epsilon every environment step
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            avg_reward = float(np.mean(episode_reward))
            rewards_history.append(episode_reward.tolist())
            avg_rewards_per_episode.append(avg_reward)
            total_energy_per_episode.append(current_episode_energy)
            total_power_per_episode.append(current_episode_power)
            avg_sbs_sinr_per_episode.append(
                sbs_sinr_sum / sinr_step_count if sinr_step_count > 0 else 0.0
            )

            print(f"Episode {ep}: Reward: {episode_reward} | "
                  f"Avg Reward: {avg_reward:.4f} | Epsilon: {epsilon:.4f}")

            # Energy efficiency
            total_bits = get_total_bits()
            ee_per_episode = [
                total_bits / e if e > 0 else 0.0
                for e in total_energy_per_episode
            ]

            # Plot average reward
            plt.figure()
            plt.plot(avg_rewards_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Average Reward per Episode (DDQN)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "avg_reward_per_episode_live.png"))
            plt.close()

            # Plot SINR
            plt.figure()
            plt.plot(avg_sbs_sinr_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Global SINR (dB)")
            plt.title("Global SINR per Episode (DDQN)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "sinr_per_episode_live.png"))
            plt.close()

            # Plot energy efficiency
            plt.figure()
            plt.plot(ee_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Energy Efficiency (bits/J)")
            plt.title("Energy Efficiency per Episode (DDQN)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode_live.png"))
            plt.close()

            # Plot loss
            if len(loss_history) > 0:
                plt.figure()
                plt.plot(loss_history)
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.title("Training Loss (DDQN)")
                plt.grid(True)
                plt.savefig(os.path.join(OUTPUT_DIR, "loss_live.png"))
                plt.close()

            # Save checkpoint
            if ep % 10 == 0:
                histories = {
                    "avg_rewards": avg_rewards_per_episode,
                    "energy": total_energy_per_episode,
                    "sinr": avg_sbs_sinr_per_episode,
                    "power": total_power_per_episode,
                    "loss": loss_history,
                }
                wrapper.save_all(
                    base_path=CHECKPOINT_PREFIX,
                    episode=ep,
                    epsilon=epsilon,
                    histories=histories
                )
                print(f">>> Checkpoint saved at Episode {ep}")

    finally:
        env.close()

    # Final save
    histories = {
        "avg_rewards": avg_rewards_per_episode,
        "energy": total_energy_per_episode,
        "sinr": avg_sbs_sinr_per_episode,
        "power": total_power_per_episode,
        "loss": loss_history,
    }

    wrapper.save_all(
        base_path=CHECKPOINT_PREFIX,
        episode=ep,
        epsilon=epsilon,
        histories=histories
    )
    wrapper.save_all_losses(base_path=CHECKPOINT_PREFIX)

    # Save SBS state history CSV
    csv_path = os.path.join(OUTPUT_DIR, "sbs_state_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step"] + [f"SBS_{i}_state" for i in range(N_AGENTS)])

        max_len = max(len(sbs_state_log[i]) for i in sbs_state_log) if N_AGENTS > 0 else 0

        for idx in range(max_len):
            try:
                ep_val, step_val = sbs_state_log[0][idx][:2]
                row = [ep_val, step_val] + [
                    sbs_state_log[i][idx][2]
                    for i in range(N_AGENTS)
                ]
                writer.writerow(row)
            except IndexError:
                continue


    # ========================================================
    # 9. FINAL BASELINE SIMULATION AND COMPARISON GRAPHS
    # ========================================================
    baseline_energy_per_episode = []
    baseline_power_per_episode = []
    baseline_sinr_per_episode = []

    os.environ["NS3_BASELINE"] = "1"
    print("\n[INFO] Launching baseline simulation for each episode...")

    for baseline_ep in range(1, EPISODES + 1):
        print(f"[Baseline] Running Episode {baseline_ep} with simSeed={baseline_ep}")

        baseline_env = ns3env.Ns3Env(
            port=5555,
            stepTime=0.01,
            startSim=True,
            simSeed=baseline_ep
        )

        energy, sinr, power = simulate_baseline_energy_and_sinr(
            baseline_env,
            episodes=1,
            max_steps=MAX_STEPS
        )

        baseline_energy_per_episode.extend(energy)
        baseline_sinr_per_episode.extend(sinr)
        baseline_power_per_episode.extend(power)

        baseline_env.close()

    os.environ["NS3_BASELINE"] = "0"

    total_bits = get_total_bits()

    # Avoid division by zero
    ee_rl = [
        total_bits / energy if energy > 0 else 0.0
        for energy in total_energy_per_episode
    ]

    ee_baseline = [
        total_bits / energy if energy > 0 else 0.0
        for energy in baseline_energy_per_episode
    ]

    # Save per-episode result arrays.
    np.save(os.path.join(OUTPUT_DIR, "rl_energy_per_episode.npy"), np.array(total_energy_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "baseline_energy_per_episode.npy"), np.array(baseline_energy_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "baseline_power_per_episode.npy"), np.array(baseline_power_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "baseline_sinr_per_episode.npy"), np.array(baseline_sinr_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "energy_efficiency_baseline.npy"), np.array(ee_baseline))

    np.save(os.path.join(OUTPUT_DIR, "avg_reward_per_episode.npy"), np.array(avg_rewards_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "sinr_per_episode.npy"), np.array(avg_sbs_sinr_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode.npy"), np.array(ee_rl))
    np.save(os.path.join(OUTPUT_DIR, "total_power_per_episode.npy"), np.array(total_power_per_episode))
    np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(OUTPUT_DIR, "step_rewards.npy"), np.array(step_rewards))

    # === Plot: Energy Consumption Comparison ===
    plt.figure(figsize=(10, 6))
    plt.plot(total_energy_per_episode, label="RL")
    plt.plot(baseline_energy_per_episode, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Energy (J)")
    plt.title("Energy Consumption Comparison (DDQN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_comparison.png"))
    plt.close()

    # === Plot: Power Consumption Comparison ===
    plt.figure(figsize=(10, 6))
    plt.plot(total_power_per_episode, label="RL")
    plt.plot(baseline_power_per_episode, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Power (W)")
    plt.title("Power Comparison (DDQN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "power_comparison.png"))
    plt.close()

    # === Plot: SINR Comparison ===
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(avg_sbs_sinr_per_episode, label="RL", linewidth=2)
        plt.plot(baseline_sinr_per_episode, label="Baseline", linestyle="--", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Global SINR (dB)")
        plt.title("SINR Comparison (DDQN)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "sinr_comparison.png"))
        plt.close()
    except Exception as e:
        print("Could not plot SINR comparison:", e)

    # === Plot: Energy Efficiency Comparison ===
    plt.figure(figsize=(10, 6))
    plt.plot(ee_rl, label="RL")
    plt.plot(ee_baseline, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bits/J)")
    plt.title("Energy Efficiency Comparison (DDQN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_comparison.png"))
    plt.close()

    # === Plot: Energy Efficiency Across Episodes ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ee_rl) + 1), ee_rl, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bits/J)")
    plt.title("Energy Efficiency per Episode (DDQN)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode.png"))
    plt.close()

    # === Plot: Global Average SINR Across Episodes ===
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(avg_sbs_sinr_per_episode) + 1),
        avg_sbs_sinr_per_episode,
        label="Global Average SINR"
    )
    plt.xlabel("Episode")
    plt.ylabel("Global SINR (dB)")
    plt.title("Global SINR per Episode (DDQN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sinr_per_episode.png"))
    plt.close()

    # === Plot: Average Reward Across Episodes ===
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_per_episode, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Episode (DDQN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_reward_per_episode.png"))
    plt.close()

    # === Plot: Loss Across Training Steps ===
    if len(loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss (DDQN)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
        plt.close()


    print("==== DDQN Training Finished ====")
    print(f"Final model saved in: {OUTPUT_DIR}")
