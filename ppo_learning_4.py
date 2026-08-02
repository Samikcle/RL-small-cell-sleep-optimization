import os
import csv
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
OUTPUT_DIR = "ppo_results_4_500_5"
CHECKPOINT_PREFIX = "trained_ppo_agent"


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

GAMMA = 0.95
GAE_LAMBDA = 0.90
CLIP_EPS = 0.10
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
K_EPOCHS = 4
MINIBATCH_SIZE = 128
ENTROPY_COEF = 0.001
VALUE_COEF = 1.0
GRAD_CLIP_NORM = 0.50

RESUME = True


# ============================================================
# 3. SMALL UTILS
# ============================================================
def split_obs(obs, n_agents=N_AGENTS, state_dim_per_agent=STATE_DIM_PER_AGENT):
    return [np.asarray(obs[i * state_dim_per_agent:(i + 1) * state_dim_per_agent], dtype=np.float32)
            for i in range(n_agents)]


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


def save_ddqn_style_graphs(
    output_dir,
    algorithm_label,
    avg_rewards_per_episode,
    total_energy_per_episode,
    total_power_per_episode,
    avg_sbs_sinr_per_episode,
    baseline_energy_per_episode=None,
    baseline_power_per_episode=None,
    baseline_sinr_per_episode=None,
):
    """Create the same final .npy and .png graph outputs as fast_ddqn_multiagent.py."""
    os.makedirs(output_dir, exist_ok=True)
    total_bits = get_total_bits()

    ee_per_episode = [total_bits / e if e > 0 else 0.0 for e in total_energy_per_episode]

    # Save per-episode result arrays.
    np.save(os.path.join(output_dir, "avg_reward_per_episode.npy"), np.array(avg_rewards_per_episode))
    np.save(os.path.join(output_dir, "sinr_per_episode.npy"), np.array(avg_sbs_sinr_per_episode))
    np.save(os.path.join(output_dir, "energy_efficiency_per_episode.npy"), np.array(ee_per_episode))
    np.save(os.path.join(output_dir, "rl_energy_per_episode.npy"), np.array(total_energy_per_episode))

    # Final energy efficiency across episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ee_per_episode) + 1), ee_per_episode, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bits/J)")
    plt.title("Energy Efficiency per Episode (PPO)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_efficiency_per_episode.png"))
    plt.close()

    # Final SINR across episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_sbs_sinr_per_episode) + 1), avg_sbs_sinr_per_episode, label="Global Average SINR")
    plt.xlabel("Episode")
    plt.ylabel("Global SINR (dB)")
    plt.title("Global SINR per Episode (PPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sinr_per_episode.png"))
    plt.close()

    # Final reward learning progress
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

    # Create baseline comparison graphs when baseline data is available.
    if baseline_energy_per_episode is None:
        return

    baseline_energy_per_episode = list(baseline_energy_per_episode)
    baseline_power_per_episode = list(baseline_power_per_episode or [])
    baseline_sinr_per_episode = list(baseline_sinr_per_episode or [])
    ee_baseline = [total_bits / e if e > 0 else 0.0 for e in baseline_energy_per_episode]

    np.save(os.path.join(output_dir, "baseline_energy_per_episode.npy"), np.array(baseline_energy_per_episode))
    np.save(os.path.join(output_dir, "baseline_power_per_episode.npy"), np.array(baseline_power_per_episode))
    np.save(os.path.join(output_dir, "baseline_sinr_per_episode.npy"), np.array(baseline_sinr_per_episode))
    np.save(os.path.join(output_dir, "energy_efficiency_baseline.npy"), np.array(ee_baseline))

    plt.figure(figsize=(10, 6))
    plt.plot(total_energy_per_episode, label="RL")
    plt.plot(baseline_energy_per_episode, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Energy (J)")
    plt.title("Energy Consumption Comparison (PPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(total_power_per_episode, label="RL")
    plt.plot(baseline_power_per_episode, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Power (W)")
    plt.title("Power Comparison (PPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_sbs_sinr_per_episode, label="RL", linewidth=2)
    plt.plot(baseline_sinr_per_episode, label="Baseline", linestyle="--", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Global SINR (dB)")
    plt.title("SINR Comparison (PPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sinr_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ee_per_episode, label="RL")
    plt.plot(ee_baseline, label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bits/J)")
    plt.title("Energy Efficiency Comparison (PPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_efficiency_comparison.png"))
    plt.close()


# ============================================================
# 4. TRAJECTORY BUFFER
# ============================================================
class TrajectoryBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def store(self, state, action, reward, next_state, done, log_prob):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))

    def __len__(self):
        return len(self.states)


# ============================================================
# 5. PPO AGENT
# ============================================================
class PPOActorCriticAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id

        self.buffer = TrajectoryBuffer()

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=CRITIC_LR)

        self.actor_loss_history = []
        self.critic_loss_history = []
        self.total_loss_history = []

    def _build_actor(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.action_size, activation="softmax")
        ])
        return model

    def _build_critic(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="linear")
        ])
        return model

    def _apply_action_mask_numpy(self, probs, is_transitioning):
        probs = np.asarray(probs, dtype=np.float64)

        if is_transitioning:
            masked = np.zeros_like(probs)
            masked[0] = 1.0
            return masked

        probs_sum = np.sum(probs)
        if probs_sum <= 0 or np.isnan(probs_sum):
            return np.ones(self.action_size, dtype=np.float64) / self.action_size

        probs = probs / probs_sum
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs)
        return probs

    def _apply_action_mask_tf(self, probs, states):
        """
        If state[-1] == 1, force action 0.
        Otherwise use the actor's softmax output.
        """
        transition_flag = tf.cast(tf.greater(states[:, -1], 0.5), tf.float32)
        forced = tf.concat(
            [tf.ones((tf.shape(probs)[0], 1), dtype=tf.float32),
             tf.zeros((tf.shape(probs)[0], self.action_size - 1), dtype=tf.float32)],
            axis=1
        )
        probs = transition_flag[:, None] * forced + (1.0 - transition_flag[:, None]) * probs
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        return probs

    def act(self, state):
        """
        Stochastic action selection from π(a|s; θ).
        No epsilon-greedy.
        """
        state = np.asarray(state, dtype=np.float32)
        is_transitioning = bool(state[-1] > 0.5)

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        probs = self.actor(state_tensor, training=False).numpy()[0]
        probs = self._apply_action_mask_numpy(probs, is_transitioning)

        action = np.random.choice(self.action_size, p=probs)
        action_prob = probs[action]
        log_prob = np.log(action_prob + 1e-8)

        value = float(self.critic(state_tensor, training=False).numpy()[0, 0])

        return action, log_prob, value

    def _compute_gae(self, rewards, values, next_values, dones):
        """
        GAE:
        delta_t = r_t + γ V(s_{t+1}) - V(s_t)
        A_t = delta_t + γ λ A_{t+1}
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        if len(self.buffer) == 0:
            return None, None, None

        states = np.asarray(self.buffer.states, dtype=np.float32)
        actions = np.asarray(self.buffer.actions, dtype=np.int32)
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        next_states = np.asarray(self.buffer.next_states, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        old_log_probs = np.asarray(self.buffer.log_probs, dtype=np.float32)

        values = self.critic(states, training=False).numpy().squeeze(-1)
        next_values = self.critic(next_states, training=False).numpy().squeeze(-1)

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(actions, dtype=tf.int32),
                tf.convert_to_tensor(old_log_probs, dtype=tf.float32),
                tf.convert_to_tensor(returns, dtype=tf.float32),
                tf.convert_to_tensor(advantages, dtype=tf.float32),
            )
        )

        dataset = dataset.shuffle(buffer_size=min(len(states), 1024), reshuffle_each_iteration=True)
        dataset = dataset.batch(min(MINIBATCH_SIZE, len(states)))

        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_total_loss = 0.0

        for _ in range(K_EPOCHS):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataset:
                with tf.GradientTape(persistent=True) as tape:
                    probs = self.actor(batch_states, training=True)
                    probs = self._apply_action_mask_tf(probs, batch_states)

                    action_one_hot = tf.one_hot(batch_actions, depth=self.action_size)
                    action_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
                    new_log_probs = tf.math.log(action_probs + 1e-8)

                    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)

                    ratios = tf.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = tf.clip_by_value(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - ENTROPY_COEF * tf.reduce_mean(entropy)

                    values_pred = tf.squeeze(self.critic(batch_states, training=True), axis=1)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values_pred))

                    total_loss = actor_loss + VALUE_COEF * critic_loss

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

                actor_grads, _ = tf.clip_by_global_norm(actor_grads, GRAD_CLIP_NORM)
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, GRAD_CLIP_NORM)

                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                last_actor_loss = float(actor_loss.numpy())
                last_critic_loss = float(critic_loss.numpy())
                last_total_loss = float(total_loss.numpy())

                del tape

        self.actor_loss_history.append(last_actor_loss)
        self.critic_loss_history.append(last_critic_loss)
        self.total_loss_history.append(last_total_loss)

        self.buffer.clear()
        return last_actor_loss, last_critic_loss, last_total_loss

    def save(self, path_prefix, episode):
        self.actor.save(f"{path_prefix}_actor.keras")
        self.critic.save(f"{path_prefix}_critic.keras")
        meta = {"last_episode": episode}
        np.save(f"{path_prefix}_meta.npy", meta)

    def load(self, path_prefix):
        self.actor = keras.models.load_model(f"{path_prefix}_actor.keras")
        self.critic = keras.models.load_model(f"{path_prefix}_critic.keras")
        meta = np.load(f"{path_prefix}_meta.npy", allow_pickle=True).item()
        return meta.get("last_episode", 0)

    def save_loss_history(self, path_prefix):
        np.save(f"{path_prefix}_actor_loss.npy", np.array(self.actor_loss_history))
        np.save(f"{path_prefix}_critic_loss.npy", np.array(self.critic_loss_history))
        np.save(f"{path_prefix}_total_loss.npy", np.array(self.total_loss_history))


# ============================================================
# 6. MULTI-AGENT PPO WRAPPER
# ============================================================
class MultiAgentPPOWrapper:
    def __init__(self, n_agents, state_dim_per_agent, action_dim):
        self.n_agents = n_agents
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim = action_dim
        self.agents = [PPOActorCriticAgent(state_dim_per_agent, action_dim, i)
                       for i in range(n_agents)]

    def split_obs(self, obs):
        return split_obs(obs, self.n_agents, self.state_dim_per_agent)

    def act(self, agent_states):
        actions = []
        log_probs = []
        values = []
        for i, agent in enumerate(self.agents):
            action, log_prob, value = agent.act(agent_states[i])
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        return actions, log_probs, values

    def remember(self, agent_states, actions, rewards, next_agent_states, dones, log_probs):
        for i, agent in enumerate(self.agents):
            agent.buffer.store(
                state=agent_states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_agent_states[i],
                done=dones[i],
                log_prob=log_probs[i]
            )

    def update(self):
        actor_losses = []
        critic_losses = []
        total_losses = []

        for agent in self.agents:
            a_loss, c_loss, t_loss = agent.update()
            if a_loss is not None:
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                total_losses.append(t_loss)

        if len(actor_losses) == 0:
            return None, None, None

        return float(np.mean(actor_losses)), float(np.mean(critic_losses)), float(np.mean(total_losses))

    def save_all(self, base_path="agent", episode=0, histories=None):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            agent.save(full_path, episode)

        if histories:
            for name, data in histories.items():
                save_path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
                np.save(save_path, np.array(data))

    def load_all(self, base_path="agent"):
        last_ep = 0
        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            last_ep = agent.load(full_path)

        hist_data = {}
        for name in ["avg_rewards", "energy", "sinr", "power", "actor_loss", "critic_loss", "total_loss"]:
            path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
            if os.path.exists(path):
                hist_data[name] = list(np.load(path))

        return last_ep, hist_data

    def save_all_losses(self, base_path="agent"):
        for i, agent in enumerate(self.agents):
            full_path = os.path.join(OUTPUT_DIR, f"{base_path}_{i}")
            agent.save_loss_history(full_path)


# ============================================================
# 7. BASELINE SIMULATION
# ============================================================
def simulate_baseline_energy_and_sinr(env, episodes, max_steps):
    baseline_energy_per_episode = []
    baseline_sinr_per_episode = []
    baseline_power_per_episode = []

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

        avg_sinr = sinr_sum / sinr_steps if sinr_steps else 0.0
        baseline_energy_per_episode.append(total_energy)
        baseline_power_per_episode.append(total_power)
        baseline_sinr_per_episode.append(avg_sinr)

    return baseline_energy_per_episode, baseline_sinr_per_episode, baseline_power_per_episode


# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wrapper = MultiAgentPPOWrapper(N_AGENTS, STATE_DIM_PER_AGENT, ACTION_DIM)

    avg_rewards_per_episode = []
    total_energy_per_episode = []
    total_power_per_episode = []
    avg_sbs_sinr_per_episode = []
    rewards_history = []
    step_rewards = []
    actor_loss_history = []
    critic_loss_history = []
    total_loss_history = []
    sbs_state_log = {i: [] for i in range(N_AGENTS)}

    start_episode = 1
    ep = 0

    expected_meta = os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_0_meta.npy")
    if RESUME and os.path.exists(expected_meta):
        print(f">>> SUCCESS: Checkpoint found in {OUTPUT_DIR}. Resuming PPO...")
        last_saved_episode, histories = wrapper.load_all(CHECKPOINT_PREFIX)
        start_episode = last_saved_episode + 1
        ep = last_saved_episode

        avg_rewards_per_episode = histories.get("avg_rewards", [])
        total_energy_per_episode = histories.get("energy", [])
        avg_sbs_sinr_per_episode = histories.get("sinr", [])
        total_power_per_episode = histories.get("power", [])
        actor_loss_history = histories.get("actor_loss", [])
        critic_loss_history = histories.get("critic_loss", [])
        total_loss_history = histories.get("total_loss", [])

        print(f">>> Resuming from Episode {start_episode}.")
    else:
        print(f">>> NOTICE: No checkpoint in {OUTPUT_DIR}. Starting fresh.")

    print("==== PPO Training Start ====")

    env = ns3env.Ns3Env(port=5556, stepTime=0.01, startSim=True, simSeed=1)

    try:
        for ep in range(start_episode, EPISODES + 1):
            env.simSeed = ep
            obs = env.reset()

            agent_states = wrapper.split_obs(obs)
            done = False
            episode_rewards = np.zeros(N_AGENTS, dtype=np.float32)
            step_count = 0

            current_episode_energy = 0.0
            current_episode_power = 0.0
            sbs_sinr_sum = 0.0
            sinr_step_count = 0

            while not done and step_count < MAX_STEPS:
                print("--------------------")
                print(f"Step {step_count + 1} (Episode {ep})")
                print("--------------------")

                actions, log_probs, values = wrapper.act(agent_states)

                next_obs, reward, done, info = env.step(np.array(actions, dtype=np.uint32))
                next_agent_states = wrapper.split_obs(next_obs)

                for i in range(N_AGENTS):
                    current_state = int(next_obs[i * STATE_DIM_PER_AGENT + 1])
                    sbs_state_log[i].append((ep, step_count, current_state))

                info_parts = parse_info(info)

                print(f"Step {step_count + 1} | Reward: {reward} | Done: {done} | "
                      f"Info: {info_parts} | Next Obs: {next_obs}")

                current_episode_energy = safe_float(info_parts, "total_energy", current_episode_energy)
                current_episode_power = safe_float(info_parts, "total_power", current_episode_power)
                global_sinr_value = safe_float(info_parts, "global_sinr", 0.0)

                sinr_step_count += 1
                sbs_sinr_sum += global_sinr_value

                print(f"Step result: next_obs={next_obs}, reward={reward}, done={done}, info={info}")

                rewards = [reward] * N_AGENTS if isinstance(reward, (float, int, np.floating)) else list(reward)
                dones = [done] * N_AGENTS

                wrapper.remember(
                    agent_states=agent_states,
                    actions=actions,
                    rewards=rewards,
                    next_agent_states=next_agent_states,
                    dones=dones,
                    log_probs=log_probs
                )

                agent_states = next_agent_states
                episode_rewards += np.array(rewards, dtype=np.float32)
                step_count += 1

            actor_loss, critic_loss, total_loss = wrapper.update()
            if actor_loss is not None:
                actor_loss_history.append(actor_loss)
                critic_loss_history.append(critic_loss)
                total_loss_history.append(total_loss)

            print(f"Episode {ep}: Reward: {episode_rewards} | "
                  f"Actor Loss: {actor_loss if actor_loss is not None else 'N/A'} | "
                  f"Critic Loss: {critic_loss if critic_loss is not None else 'N/A'}")

            avg_reward = float(np.mean(episode_rewards))
            rewards_history.append(episode_rewards.tolist())
            avg_rewards_per_episode.append(avg_reward)
            total_energy_per_episode.append(current_episode_energy)
            total_power_per_episode.append(current_episode_power)
            avg_sbs_sinr_per_episode.append(sbs_sinr_sum / sinr_step_count if sinr_step_count > 0 else 0.0)

            packets_per_ue = SIM_TIME / PACKET_INTERVAL
            total_packets = packets_per_ue * NUM_UES
            total_bits = total_packets * PACKET_SIZE_BYTES * 8
            ee_per_episode = [total_bits / e if e > 0 else 0.0 for e in total_energy_per_episode]

            # Live plots
            plt.figure()
            plt.plot(avg_rewards_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Average Reward per Episode (PPO)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "avg_reward_per_episode_live.png"))
            plt.close()

            plt.figure()
            plt.plot(avg_sbs_sinr_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Global SINR (dB)")
            plt.title("Global SINR per Episode (PPO)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "sinr_per_episode_live.png"))
            plt.close()

            plt.figure()
            plt.plot(ee_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Energy Efficiency (bits/J)")
            plt.title("Energy Efficiency per Episode (PPO)")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode_live.png"))
            plt.close()

            if ep % 10 == 0:
                hist_to_save = {
                    "avg_rewards": avg_rewards_per_episode,
                    "energy": total_energy_per_episode,
                    "sinr": avg_sbs_sinr_per_episode,
                    "power": total_power_per_episode,
                    "actor_loss": actor_loss_history,
                    "critic_loss": critic_loss_history,
                    "total_loss": total_loss_history,
                }
                wrapper.save_all(CHECKPOINT_PREFIX, ep, histories=hist_to_save)
                print(f"Checkpoint saved at Episode {ep}")

        env.close()

        wrapper.save_all(CHECKPOINT_PREFIX, ep, histories={
            "avg_rewards": avg_rewards_per_episode,
            "energy": total_energy_per_episode,
            "sinr": avg_sbs_sinr_per_episode,
            "power": total_power_per_episode,
            "actor_loss": actor_loss_history,
            "critic_loss": critic_loss_history,
            "total_loss": total_loss_history,
        })
        print("Final model saved.")

        wrapper.save_all_losses(CHECKPOINT_PREFIX)
        print("Loss histories saved.")

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

        # Run the baseline and create final comparison graphs.
        os.environ["NS3_BASELINE"] = "1"
        print("\n[INFO] Launching baseline simulation for each episode...")
        baseline_energy_per_episode = []
        baseline_power_per_episode = []
        baseline_sinr_per_episode = []

        num_episodes_to_plot = len(total_energy_per_episode)
        for b_ep in range(1, num_episodes_to_plot + 1):
            print(f"[Baseline] Running Episode {b_ep} with simSeed={b_ep}")
            env = ns3env.Ns3Env(port=5556, stepTime=0.01, startSim=True, simSeed=b_ep)
            energy, sinr, power = simulate_baseline_energy_and_sinr(env, 1, MAX_STEPS)
            baseline_energy_per_episode.extend(energy)
            baseline_power_per_episode.extend(power)
            baseline_sinr_per_episode.extend(sinr)
            env.close()

        os.environ["NS3_BASELINE"] = "0"

        save_ddqn_style_graphs(
            output_dir=OUTPUT_DIR,
            algorithm_label="PPO",
            avg_rewards_per_episode=avg_rewards_per_episode,
            total_energy_per_episode=total_energy_per_episode,
            total_power_per_episode=total_power_per_episode,
            avg_sbs_sinr_per_episode=avg_sbs_sinr_per_episode,
            baseline_energy_per_episode=baseline_energy_per_episode,
            baseline_power_per_episode=baseline_power_per_episode,
            baseline_sinr_per_episode=baseline_sinr_per_episode,
        )
        print("DDQN-style PPO plots saved successfully.")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        wrapper.save_all(CHECKPOINT_PREFIX, ep, histories={
            "avg_rewards": avg_rewards_per_episode,
            "energy": total_energy_per_episode,
            "sinr": avg_sbs_sinr_per_episode,
            "power": total_power_per_episode,
            "actor_loss": actor_loss_history,
            "critic_loss": critic_loss_history,
            "total_loss": total_loss_history,
        })
        wrapper.save_all_losses(CHECKPOINT_PREFIX)
        env.close()
        print("Saved and exited cleanly.")
