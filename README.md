# Reinforcement Learning for Small-Cell Sleep Strategy Optimization

This project compares three reinforcement learning algorithms for optimizing the sleep states of small-cell base stations in an ns-3 simulation:

- Double Deep Q-Network with Prioritized Experience Replay (DDQN-PER)
- Proximal Policy Optimization (PPO)
- Tabular Q-Learning

The algorithms aim to reduce small-cell energy consumption while maintaining acceptable network signal quality, represented by the global Signal-to-Interference-plus-Noise Ratio (SINR). Training results are recorded per episode, while a separate evaluation script compares all trained models against an always-ACTIVE baseline using the same simulation seeds.

## Project Files

| File | Purpose |
|---|---|
| `scratch.cc` | Main ns-3 LTE/OpenGym simulation environment. It creates the macro base station, five small-cell base stations, 30 user devices, traffic, mobility, handover behavior, observations, rewards, and action handling. |
| `smallCellEnergyModel.h` | Custom ns-3 energy model for the small-cell base stations. It defines the ACTIVE, SM1, SM2, and SM3 states, their power consumption, activation delays, transition behavior, and energy accounting. |
| `fast_ddqn_multiagent_c_5.py` | Multi-agent DDQN-PER training script. It trains one DDQN agent for each of the five small cells and saves checkpoints, training metrics, baseline results, and graphs. |
| `ppo_learning_4.py` | Multi-agent PPO training script. It uses actor-critic networks, generalized advantage estimation, clipped policy updates, and per-agent checkpoints. |
| `tabular_q_learning_c_10.py` | Multi-agent tabular Q-learning training script. It discretizes each small cell's observation and stores learned action values in a Q-table. |
| `test_all_models_checkpoint.py` | Evaluates DDQN, PPO, Q-learning, and the always-ACTIVE baseline using common test seeds. It supports checkpointing and resuming interrupted evaluations. |
| `plot_training_percentages_and_sinr_thresholds.py` | Post-processes each algorithm's training outputs separately. It adds SINR quality thresholds and converts energy and energy efficiency to percentages relative to that algorithm's baseline. |
| `plot_sinr_thresholds_percentages.py` | Post-processes the combined testing results. It adds SINR quality thresholds and produces energy and efficiency comparisons relative to the baseline. |
| `convert_rl_npy_to_csv.py` | Converts saved NumPy result files into individual CSV files and a combined episode-level comparison table. |

## Simulation Structure

The active simulation configuration uses:

- 1 macro base station
- 5 small-cell base stations
- 30 user devices, corresponding to 6 user devices per small cell in the initial configuration
- 10-second simulation duration
- 0.01-second reinforcement-learning step interval
- Four available small-cell actions/states:
  - `0`: ACTIVE
  - `1`: SM1
  - `2`: SM2
  - `3`: SM3

The small-cell energy model uses the following total-power levels relative to ACTIVE operation:

| State | Power Consumption | (De)activation Duration |
|---|---:|---:|
| ACTIVE | 100% | 35.5 µs |
| SM1 | 69% | 0.5 ms |
| SM2 | 50% | 5 ms |
| SM3 | 29% | 0.5 s |

A sleeping small cell requires a state-dependent activation delay before returning to ACTIVE operation.

## Agent Observation

Each reinforcement-learning agent receives five values for its associated small cell:

```text
[active_ues, state, power, global_sinr, transitioning]
```

With five agents and five values per agent, the complete observation contains 25 values.

## Recorded Metrics

### Training

Each algorithm records episode-level metrics including:

- Average reward
- Global SINR
- Total energy consumption
- Total power consumption where available
- Energy efficiency
- Algorithm-specific learning information, such as DDQN loss, PPO actor/critic losses, or Q-learning temporal-difference error

Each training script also runs or stores an always-ACTIVE baseline for comparison.

### Testing

The common evaluation script records the following for DDQN, PPO, Q-learning, and the baseline:

- Average SINR
- Total energy consumption
- Energy efficiency
- Total power
- Average reward
- Average number of active user devices
- Small-cell states across simulation steps

The default evaluation uses 150 randomly generated test seeds.

## SINR Quality Categories

The plotting scripts use the following SINR categories:

| Category | SINR range |
|---|---:|
| Poor | Below 0 dB |
| Fair | 0 dB to below 13 dB |
| Good | 13 dB to below 20 dB |
| Excellent | 20 dB or above |

These boundaries are displayed as magenta threshold lines in the post-processed SINR graphs.

## Requirements

The project requires an ns-3 installation configured with ns3-gym/OpenGym support. The Python environment requires packages compatible with the training scripts, including:

```text
numpy
matplotlib
tensorflow
ns3gym
```

The scripts also use Python standard-library modules such as `argparse`, `csv`, `pickle`, `pathlib`, and `importlib`.

The exact ns-3 and ns3-gym installation commands depend on the versions and directory structure used by the local simulation environment.

## Preparing the Environment

1. Install ns-3 (version 3.40 is preferred)
2. Install ns3-gym
3. Build ns-3
4. Setup Python virtual environment
5. Install required python packages
6. Place all files in this repository into ns-3/scratch/

## Expected Project Layout

```text
ns-3/
└── scratch/
    ├── scratch.cc
    ├── smallCellEnergyModel.h
    ├── fast_ddqn_multiagent_c_5.py
    ├── ppo_learning_4.py
    ├── tabular_q_learning_c_10.py
    ├── test_all_models_checkpoint.py
    ├── plot_training_percentages_and_sinr_thresholds.py
    ├── plot_sinr_thresholds_percentages.py
    ├── convert_rl_npy_to_csv.py
    ├── ddqn_per_results_c_5_500_2/
    ├── ppo_results_4_500_5/
    ├── qlearning_results_c_10_500_2/
    ├── all_model_test_results/
    └── csv_results/
```

## Running the Simulation Environment

The Python training and testing scripts create `ns3gym.Ns3Env` with `startSim=True`, so ns3-gym starts the configured ns-3 simulation when an environment is created.

The C++ environment accepts the following command-line parameters:

```text
--openGymPort
--simSeed
--simTime
--stepTime
```

Default values in the C++ environment are:

```text
openGymPort = 5555
simSeed     = 1
simTime     = 10 seconds
stepTime    = 0.01 seconds
```

Ensure that the ns3-gym configuration launches the correct ns-3 program corresponding to `scratch.cc`.

## Training the Models

Run the algorithms separately unless the ns-3 setup is configured to support concurrent processes and separate ports.

### DDQN-PER

```bash
python3 fast_ddqn_multiagent_c_5.py
```

Default output directory:

```text
ddqn_per_results_c_5_500_2/
```

Default ns3-gym port:

```text
5555
```

### PPO

```bash
python3 ppo_learning_4.py
```

Default output directory:

```text
ppo_results_4_500_5/
```

Default ns3-gym port:

```text
5556
```

### Tabular Q-Learning

```bash
python3 tabular_q_learning_c_10.py
```

Default output directory:

```text
qlearning_results_c_10_500_2/
```

Default ns3-gym port:

```text
5557
```

All three scripts are configured for 500 training episodes and a maximum of 1,000 steps per episode. `RESUME = True` enables checkpoint loading when compatible checkpoints already exist.

To begin a fully new training run, move or delete the corresponding checkpoint and result files, or change the resume setting after confirming that no existing results are needed.

## Training Output Directories

The training scripts save combinations of:

- Model checkpoints
- `.npy` metric arrays
- `.csv` logs
- Per-episode graphs
- Baseline comparison graphs

The expected primary result folders are:

```text
ddqn_per_results_c_5_500_2/
ppo_results_4_500_5/
qlearning_results_c_10_500_2/
```

Do not mix checkpoints from different hyperparameter configurations unless their network structures, action spaces, and state structures are compatible.

## Evaluating All Models

After all models have been trained and their checkpoint files are available, run:

```bash
python3 test_all_models_checkpoint.py
```

The evaluation script loads the three trained models and tests them with the same randomly generated seeds. It uses deterministic action selection:

- DDQN: greedy action selection with epsilon set to zero
- PPO: action with the highest valid policy probability
- Q-learning: greedy Q-table action selection with epsilon set to zero
- Baseline: all small cells remain ACTIVE

Default output directory:

```text
all_model_test_results/
```

Useful options include:

```bash
python3 test_all_models_checkpoint.py --episodes 150
python3 test_all_models_checkpoint.py --checkpoint-interval 10
python3 test_all_models_checkpoint.py --restart
python3 test_all_models_checkpoint.py --port 5555
```

The script automatically preserves completed results in:

```text
all_model_test_results/evaluation_checkpoint.pkl
```

By default, an interrupted evaluation resumes when the saved configuration and test seeds match the requested run. Use `--restart` to remove the existing evaluation checkpoint and start again.

## Post-Processing Training Results

To create separate percentage-based training graphs for each algorithm, run:

```bash
python3 plot_training_percentages_and_sinr_thresholds.py
```

The script reads the three configured training output folders and creates this subdirectory inside each one:

```text
postprocessed_percent_thresholds_original_style/
```

Outputs include:

- SINR comparison with quality thresholds
- Energy consumption as a percentage of baseline
- Energy efficiency as a percentage of baseline
- CSV data used in the post-processed graphs

A different output subdirectory can be selected with:

```bash
python3 plot_training_percentages_and_sinr_thresholds.py \
  --output-subdir my_training_plots
```

## Post-Processing Combined Test Results

After running the common evaluation, use:

```bash
python3 plot_sinr_thresholds_percentages.py
```

Default input directory:

```text
all_model_test_results/
```

Default output subdirectory:

```text
all_model_test_results/threshold_percentage_plots_magenta_threshold/
```

The script generates:

- SINR comparison with quality thresholds
- Average SINR bar chart
- Energy consumption as a percentage of baseline
- Energy saving relative to baseline
- Energy efficiency as a percentage of baseline
- Average percentage comparison charts
- CSV summaries

Custom paths can be provided using:

```bash
python3 plot_sinr_thresholds_percentages.py \
  --input-dir all_model_test_results \
  --output-subdir custom_test_plots
```

## Converting NumPy Results to CSV

To convert the saved `.npy` files into CSV format, run:

```bash
python3 convert_rl_npy_to_csv.py
```

Default output directory:

```text
csv_results/
```

The converter creates:

```text
csv_results/DDQN/
csv_results/PPO/
csv_results/Q_Learning/
csv_results/all_models_combined_metrics.csv
csv_results/conversion_log.csv
```

Optional arguments:

```bash
python3 convert_rl_npy_to_csv.py --base-dir /path/to/project
python3 convert_rl_npy_to_csv.py --output-dir my_csv_results
python3 convert_rl_npy_to_csv.py --include-model-artifacts
```

The `--include-model-artifacts` option also converts files such as Q-tables and metadata arrays, which are normally excluded because they are not standard result tables.

## Recommended Execution Order

```text
1. Configure and build the ns-3/ns3-gym environment.
2. Train DDQN-PER.
3. Train PPO.
4. Train tabular Q-learning.
5. Confirm that all trained checkpoints are available.
6. Run the common testing script.
7. Generate the separate training graphs.
8. Generate the combined testing graphs.
9. Convert NumPy results to CSV for statistical analysis or reporting.
```

## Reproducibility Notes

- Training episodes use changing simulation seeds based on the episode number.
- The testing script generates a fixed set of random test seeds using its configured seed, allowing all algorithms to be evaluated under the same scenarios.
- Keep the environment configuration, reward function, user traffic, topology, state representation, and testing seeds identical when comparing algorithms.
- Record the exact hyperparameters and output-folder names associated with each final model.
- Avoid evaluating a checkpoint with a different action dimension or observation structure from the one used during training.
- The energy-efficiency calculation assumes the configured packet size, packet interval, simulation duration, and number of user devices remain unchanged.

## Important Configuration Consistency

Before running a final experiment, verify that the following values agree across the C++ environment and Python scripts:

```text
Number of small cells
Number of user devices
Simulation duration
RL step interval
Action dimension
Observation dimension
Packet interval
Packet size
Checkpoint names
Result-folder names
ns3-gym port
```

Changing these values in only one file can produce incompatible checkpoints, incorrect energy-efficiency calculations, failed model loading, or invalid comparisons.

## Research Use

The resulting data can support comparisons of:

- Reward convergence
- SINR preservation
- Energy reduction relative to the always-ACTIVE baseline
- Energy-efficiency improvement
- Training stability
- Generalization across unseen simulation seeds
- Trade-offs between network quality and energy saving

For a fair research comparison, report the same episode range, smoothing method, testing seeds, baseline definition, and metric formulas for all three algorithms.
