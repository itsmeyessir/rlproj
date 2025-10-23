# LRT-2 Reinforcement Learning Project: 'Dagdag o Lapad'

This project explores the application of Reinforcement Learning (RL) to optimize the operations of Manila's Light Rail Transit Line 2 (LRT-2). The goal is to train intelligent agents that can dynamically decide whether to add more train carriages ('Dagdag') or widen existing ones ('Lapad') to maximize profitability and efficiency.

Three different RL agents are implemented and compared:
*   **Q-Learning**
*   **Monte Carlo**
*   **Actor-Critic**

## Project Structure

```
/rlproj
├── agents/             # RL agent implementations (QLearning, MonteCarlo, ActorCritic)
├── configs/            # Hyperparameter configurations for each agent
├── environment/        # The custom LRT-2 environment (lrt_env.py)
├── figures/            # Output plots from the evaluation script
├── models/             # Saved model policies for each trained agent
├── runs/               # TensorBoard logs for each training run
├── training/           # Scripts for training the agents
├── evaluate_agents.py  # Script to evaluate trained agents
├── main_demo.py        # Interactive Pygame demo to visualize agent performance
├── plot_results.py     # Script to generate plots from evaluation results
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rlproj
pip install -r requirements.txt
```

### 2. Training the Agents

To train all the agents with their different hyperparameter configurations, run the `train_agent.py` script. This will train all 9 model variations (3 agents x 3 configs each).

```bash
python training/train_agent.py
```

This will generate:
*   **TensorBoard logs** in the `runs/` directory.
*   **Saved model policies** in the `models/` directory.

### 3. Evaluating the Agents

After training, you can evaluate the performance of the saved models:

```bash
python evaluate_agents.py
```

This script will run each trained model for a set number of episodes and save the results to `evaluation_results.csv`.

### 4. Visualizing the Results

To generate plots comparing the performance of the different agents and configurations, run:

```bash
python plot_results.py
```

This will save the following plots in the `figures/` directory:
*   `avg_config_cost.png`: Average configuration cost per episode.
*   `score_distribution.png`: Score distribution and consistency.
*   `efficiency.png`: Reward per unit cost.
*   `action_distribution.png`: Action distribution comparison.

### 5. Interactive Demo

To see the best-performing agents in action, run the interactive Pygame demo:

```bash
python main_demo.py
```

This will open a window where you can select which agent to watch or even play the game yourself.

## The Environment: `Lrt2Env`

The custom environment `lrt_env.py` simulates the LRT-2 system. The key aspects are:

*   **State:** `(current_station_idx, num_carriages, carriage_width_level, direction)`
*   **Actions:**
    *   `0`: 'Dagdag' (Add a carriage)
    *   `1`: 'Lapad' (Widen carriages)
    *   `2`: No action
*   **Reward:** The reward function is based on the profit from passengers served, minus the cost of adding/widening carriages and penalties for unmet demand.
*   **Episode:** An episode consists of a full trip from one end of the line to the other (e.g., Recto to Antipolo).

## Agents

*   **Q-Learning:** A model-free, off-policy TD (Temporal-Difference) learning algorithm.
*   **Monte Carlo:** A model-free, on-policy algorithm that learns from complete episodes.
*   **Actor-Critic:** A policy gradient method that has two components: an 'actor' that controls the agent's behavior and a 'critic' that measures how good the taken action is.