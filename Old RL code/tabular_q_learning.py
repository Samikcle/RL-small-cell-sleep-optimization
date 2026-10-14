import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import random
from ns3gym import ns3env
import matplotlib.pyplot as plt
import os
import csv

# --- 1. GLOBAL PATH CONFIGURATION ---
OUTPUT_DIR = "qlearning_results_1"
CHECKPOINT_PREFIX = "trained_q_agent"

# --- 2. Hyperparameters & Env Params ---
EPISODES = 150 
MAX_STEPS = 1000
N_AGENTS = 5
ACTION_DIM = 4 

NUM_UES = 30
SIM_TIME = 10
PACKET_INTERVAL = 0.02778
PACKET_SIZE_BYTES = 1400

ALPHA = 0.1     
GAMMA = 0.98    
EPSILON = 1.0   
EPS_DECAY = 0.995
EPS_MIN = 0.05

# --- 3. Discretization Bins ---
UE_BINS = np.array([0, 2, 5, 8])          
SINR_BINS = np.array([0, 5, 10, 15])      
STATE_DIM_DISCRETE = (len(UE_BINS)+1) * (len(SINR_BINS)+1)

class TabularQLearningAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.q_table = np.zeros((STATE_DIM_DISCRETE, ACTION_DIM))

    def discretize(self, obs):
        ue_idx = np.digitize(obs[0], UE_BINS)
        sinr_idx = np.digitize(obs[3], SINR_BINS)
        return ue_idx * (len(SINR_BINS) + 1) + sinr_idx

    def act(self, state_idx, is_transitioning, current_epsilon):
        if is_transitioning: return 0  
        if np.random.rand() < current_epsilon:
            return np.random.randint(ACTION_DIM)
        return np.argmax(self.q_table[state_idx])

    def update(self, s, a, r, s_next):
        best_next_a = np.max(self.q_table[s_next])
        td_target = r + GAMMA * best_next_a
        self.q_table[s, a] += ALPHA * (td_target - self.q_table[s, a])

    def save(self, path_prefix, episode, current_epsilon):
        np.save(f"{path_prefix}_qtable.npy", self.q_table)
        meta = {"epsilon": current_epsilon, "last_episode": episode}
        np.save(f"{path_prefix}_meta.npy", meta)

    def load(self, path_prefix):
        self.q_table = np.load(f"{path_prefix}_qtable.npy")
        meta = np.load(f"{path_prefix}_meta.npy", allow_pickle=True).item()
        return meta.get("epsilon", 1.0), meta.get("last_episode", 0)

class MultiAgentWrapper:
    def __init__(self, n_agents):
        self.agents = [TabularQLearningAgent(i) for i in range(n_agents)]

    def save_all(self, base_path, episode, current_epsilon, histories=None):
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(OUTPUT_DIR, f"{base_path}_{i}"), episode, current_epsilon)
        if histories:
            for name, data in histories.items():
                np.save(os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy"), np.array(data))

    def load_all(self, base_path):
        last_ep, current_eps = 0, 1.0
        for i, agent in enumerate(self.agents):
            current_eps, last_ep = agent.load(os.path.join(OUTPUT_DIR, f"{base_path}_{i}"))
        hist_data = {}
        for name in ["avg_rewards", "energy", "sinr", "power"]:
            path = os.path.join(OUTPUT_DIR, f"{base_path}_{name}.npy")
            if os.path.exists(path): hist_data[name] = list(np.load(path))
        return last_ep, current_eps, hist_data

def simulate_baseline_energy_and_sinr(env, episodes, max_steps):
    baseline_energy, baseline_sinr, baseline_power = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done, step_count, total_energy, total_power, sinr_sum, sinr_steps = False, 0, 0.0, 0.0, 0.0, 0
        while not done and step_count < max_steps:
            action = [0] * N_AGENTS
            _, _, done, info = env.step(np.array(action, dtype=np.uint32))
            info_str = info if isinstance(info, str) else info[0]
            info_parts = dict(item.split("=") for item in info_str.split(";") if "=" in item)
            total_energy = float(info_parts.get("total_energy", 0.0))
            total_power = float(info_parts.get("total_power", 0.0))
            sinr_sum += float(info_parts.get("global_sinr", 0.0))
            sinr_steps += 1
            step_count += 1
        baseline_energy.append(total_energy)
        baseline_power.append(total_power)
        baseline_sinr.append(sinr_sum / sinr_steps if sinr_steps else 0)
    return baseline_energy, baseline_sinr, baseline_power

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wrapper = MultiAgentWrapper(N_AGENTS)
    
    # 1. Default initializations
    avg_rewards_per_episode = []
    total_energy_per_episode = []
    total_power_per_episode = []
    avg_sbs_sinr_per_episode = []
    
    # 2. Resume Logic
    RESUME = True
    start_episode = 1
    ep = 0
    
    expected_meta = os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_0_meta.npy")
    if RESUME and os.path.exists(expected_meta):
        print(f">>> SUCCESS: Checkpoint found in {OUTPUT_DIR}. Resuming Q-Learning...")
        last_saved_ep, last_eps, histories = wrapper.load_all(CHECKPOINT_PREFIX)
        start_episode = last_saved_ep + 1
        ep = last_saved_ep
        EPSILON = last_eps
        
        avg_rewards_per_episode = histories.get("avg_rewards", [])
        total_energy_per_episode = histories.get("energy", [])
        avg_sbs_sinr_per_episode = histories.get("sinr", [])
        total_power_per_episode = histories.get("power", [])
        print(f">>> Resuming from Episode {start_episode}. Epsilon: {EPSILON:.4f}")
    else:
        print(f">>> NOTICE: No checkpoint in {OUTPUT_DIR}. Starting fresh.")

    print("==== Tabular Q-Learning Training Start ====") # Matched to DDQN
    for ep in range(start_episode, EPISODES + 1):
        env = ns3env.Ns3Env(port=5555, stepTime=0.01, startSim=True, simSeed=ep)
        obs = env.reset()
        done = False
        episode_reward = 0
        sinr_sum, step_count = 0.0, 0
        current_energy, current_power = 0.0, 0.0

        while not done and step_count < MAX_STEPS:
            # --- NEW: Step Headers (Matched to DDQN) ---
            print("--------------------")
            print(f"Step {step_count+1} (Episode {ep})")
            print("--------------------")

            agent_obs = [obs[i*5 : (i+1)*5] for i in range(N_AGENTS)]
            state_indices = [wrapper.agents[i].discretize(agent_obs[i]) for i in range(N_AGENTS)]
            transitions = [obs[i*5 + 4] for i in range(N_AGENTS)]

            actions = [wrapper.agents[i].act(state_indices[i], transitions[i], EPSILON) for i in range(N_AGENTS)]
            next_obs, reward, done, info = env.step(np.array(actions, dtype=np.uint32))
            
            # --- NEW: Robust Info Parsing & Step Printing ---
            info_str = info if isinstance(info, str) else info[0]
            info_parts = dict(item.split("=") for item in info_str.split(";") if "=" in item)
            
            print(f"Step {step_count+1} | Reward: {reward} | Done: {done} | Info: {info_parts} | Next Obs: {next_obs}")
            
            current_energy = float(info_parts.get("total_energy", 0.0))
            current_power = float(info_parts.get("total_power", 0.0))
            sinr_val = float(info_parts.get("global_sinr", 0.0))
            
            # Detailed step result print
            print(f"Step result: next_obs={next_obs}, reward={reward}, done={done}, info={info}")

            for i in range(N_AGENTS):
                next_agent_obs = next_obs[i*5 : (i+1)*5]
                next_state_idx = wrapper.agents[i].discretize(next_agent_obs)
                wrapper.agents[i].update(state_indices[i], actions[i], reward, next_state_idx)
            
            obs = next_obs
            episode_reward += reward
            sinr_sum += sinr_val
            step_count += 1

        avg_rewards_per_episode.append(episode_reward)
        total_energy_per_episode.append(current_energy)
        total_power_per_episode.append(current_power)
        avg_sbs_sinr_per_episode.append(sinr_sum / step_count if step_count > 0 else 0)
        EPSILON = max(EPS_MIN, EPSILON * EPS_DECAY)

        print(f"Episode {ep}: Reward: {episode_reward:.2f} | Epsilon: {EPSILON:.4f}")

        # Efficiency Calculation
        total_bits = (SIM_TIME / PACKET_INTERVAL) * NUM_UES * PACKET_SIZE_BYTES * 8
        ee_per_episode = [total_bits / e for e in total_energy_per_episode]

        # Live Plotting
        for name, data, title, ylabel in [
            ("avg_reward_per_episode_live.png", avg_rewards_per_episode, "Avg Reward", "Reward"),
            ("sinr_per_episode_live.png", avg_sbs_sinr_per_episode, "Global Avg SINR", "SINR (dB)"),
            ("energy_efficiency_per_episode_live.png", ee_per_episode, "Energy Efficiency", "bits/Joule")
        ]:
            plt.figure(); plt.plot(data); plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel); plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, name)); plt.close()

        if ep % 10 == 0:
            wrapper.save_all(CHECKPOINT_PREFIX, ep, EPSILON, {"avg_rewards": avg_rewards_per_episode, "energy": total_energy_per_episode, "sinr": avg_sbs_sinr_per_episode, "power": total_power_per_episode})
            print(f"Checkpoint saved at Episode {ep}")
        env.close()

    # === Baseline & Final Plots ===
    os.environ["NS3_BASELINE"] = "1"
    baseline_energy_per_episode, baseline_sinr_per_episode, baseline_power_per_episode = [], [], []
    for b_ep in range(1, EPISODES + 1):
        env = ns3env.Ns3Env(port=5555, stepTime=0.01, startSim=True, simSeed=b_ep)
        be, bs, bp = simulate_baseline_energy_and_sinr(env, 1, MAX_STEPS)
        baseline_energy_per_episode.extend(be); baseline_sinr_per_episode.extend(bs); baseline_power_per_episode.extend(bp)
        env.close()
    
    # Save Data
    for n, d in [("rl_energy", total_energy_per_episode), ("baseline_energy", baseline_energy_per_episode), ("baseline_power", baseline_power_per_episode), ("baseline_sinr", baseline_sinr_per_episode)]:
        np.save(os.path.join(OUTPUT_DIR, f"{n}_qlearning.npy"), np.array(d))

    # Final Comparison and Standalone Plots (matches DDQN set)
    metrics = [
        (total_energy_per_episode, baseline_energy_per_episode, "Energy Comparison", "Joules", "energy_comparison.png"),
        (total_power_per_episode, baseline_power_per_episode, "Power Comparison", "Watts", "power_comparison.png"),
        (avg_sbs_sinr_per_episode, baseline_sinr_per_episode, "SINR Comparison", "dB", "sinr_comparison.png"),
        ([total_bits/e for e in total_energy_per_episode], [total_bits/e for e in baseline_energy_per_episode], "EE Comparison", "bits/J", "energy_efficiency_comparison.png")
    ]
    for rl, base, title, ylabel, fname in metrics:
        plt.figure(figsize=(10,6)); plt.plot(rl, label="Q-Learning"); plt.plot(base, label="Baseline", linestyle="--"); plt.title(title); plt.legend(); plt.grid(True); plt.savefig(os.path.join(OUTPUT_DIR, fname)); plt.close()

    # Standalone final plots
    plt.figure(); plt.plot(ee_per_episode); plt.title("Final EE Across Episodes"); plt.savefig(os.path.join(OUTPUT_DIR, "energy_efficiency_per_episode.png")); plt.close()
    plt.figure(); plt.plot(avg_sbs_sinr_per_episode); plt.title("Final SINR Across Episodes"); plt.savefig(os.path.join(OUTPUT_DIR, "sinr_per_episode.png")); plt.close()
    plt.figure(); plt.plot(avg_rewards_per_episode); plt.title("Final Reward Progress"); plt.savefig(os.path.join(OUTPUT_DIR, "avg_reward_per_episode.png")); plt.close()
    
    print(f"All graphs generated in {OUTPUT_DIR}")