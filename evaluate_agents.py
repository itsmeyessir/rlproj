import os
import pandas as pd
from tqdm import tqdm

# Import our custom classes
from environment.lrt_env import Lrt2Env
from agents.monte_carlo_agent import MonteCarloAgent
from agents.q_learning_agent import QLearningAgent
from agents.actor_critic_agent import ActorCriticAgent

# --- Configuration ---
NUM_EVAL_EPISODES = 1000
MODELS_DIR = 'models'
RESULTS_FILE = 'evaluation_results.csv'

# Mapping model names to their classes
AGENT_CLASSES = {
    'MonteCarlo': MonteCarloAgent,
    'QLearning': QLearningAgent,
    'ActorCritic': ActorCriticAgent
}

def evaluate():
    env = Lrt2Env()
    all_results = []
    
    # Check if models directory exists
    if not os.path.isdir(MODELS_DIR) or not os.listdir(MODELS_DIR):
        print(f"Error: The '{MODELS_DIR}' directory is empty or does not exist.")
        print("Please train the agents first by running 'python -m training.train_agent'")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.pkl', '.pt'))]

    for model_file in tqdm(model_files, desc="Evaluating Agents"):
        parts = model_file.replace('_policy.pkl', '').replace('_policy.pt', '').split('_')
        agent_name = parts[0]
        config_name = '_'.join(parts[1:])

        agent_class = AGENT_CLASSES[agent_name]
        agent = agent_class(action_space=env.action_space)
        agent.load_policy(os.path.join(MODELS_DIR, model_file))

        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0

        for i in range(NUM_EVAL_EPISODES):
            state = env.reset()
            done = False
            
            cumulative_reward = 0
            total_cost = 0
            actions = []

            while not done:
                action = agent.choose_action(state)
                actions.append(action)
                
                # --- CHANGE: Updated cost calculation for new 'Lapad' rules ---
                action_cost = 0
                current_width_level = state[2] # Get width from state
                if action == 0:
                    action_cost = env.add_carriage_cost
                elif action == 1 and current_width_level < env.max_width_level:
                    action_cost = env.widen_carriage_cost
                total_cost += abs(action_cost)
                
                next_state, reward, done, _ = env.step(action)
                cumulative_reward += reward
                state = next_state
            
            all_results.append({
                'agent': agent_name,
                'config': config_name,
                'episode': i,
                'reward': cumulative_reward,
                'cost': total_cost,
                'actions': actions
            })
            
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nEvaluation complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate()