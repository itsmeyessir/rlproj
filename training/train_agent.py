import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import our custom classes
from environment.lrt_env import Lrt2Env
from agents.monte_carlo_agent import MonteCarloAgent
from agents.q_learning_agent import QLearningAgent
from agents.actor_critic_agent import ActorCriticAgent
from training.utils import load_config

def train(agent_class, env, config, num_episodes, log_dir, model_save_path):
    """
    A flexible function to train any of our agents.
    """
    writer = SummaryWriter(log_dir=log_dir)
    agent = agent_class(action_space=env.action_space, **config)
    
    all_rewards = []
    
    # Main Training Loop
    for episode in tqdm(range(num_episodes), desc=f"Training {log_dir.split('/')[-1]}"):
        state = env.reset()
        done = False
        episode_log = []
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # --- AGENT-SPECIFIC LEARNING ---
            # Q-Learning learns at every step.
            if isinstance(agent, QLearningAgent):
                agent.learn(state, action, reward, next_state)
            
            # Monte Carlo needs the full episode log.
            elif isinstance(agent, MonteCarloAgent):
                episode_log.append((state, action, reward))

            # Our Actor-Critic agent accumulates data and learns at the end.
            elif isinstance(agent, ActorCriticAgent):
                agent.learn(reward, done)

            cumulative_reward += reward
            state = next_state
        
        # Monte Carlo learns only at the end of the episode.
        if isinstance(agent, MonteCarloAgent):
            agent.learn(episode_log)

        # Logging
        all_rewards.append(cumulative_reward)
        writer.add_scalar('Reward/Cumulative', cumulative_reward, episode)
        if episode >= 100:
            avg_reward_100 = sum(all_rewards[-100:]) / 100
            writer.add_scalar('Reward/Running_Average_100', avg_reward_100, episode)
            
    # Save the final policy and close the writer
    agent.save_policy(model_save_path)
    writer.close()
    print(f"Training complete. Policy saved to {model_save_path}")


if __name__ == "__main__":
    # --- This script will now run ALL 9 experiments sequentially ---
    EXPERIMENTS = {
        'MonteCarlo': {
            'class': MonteCarloAgent,
            'config_path': 'configs/mc_configs.json'
        },
        'QLearning': {
            'class': QLearningAgent,
            'config_path': 'configs/ql_configs.json'
        },
        'ActorCritic': {
            'class': ActorCriticAgent,
            'config_path': 'configs/ac_configs.json'
        }
    }
    
    NUM_EPISODES = 20000 
    env = Lrt2Env()
    
    os.makedirs('runs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
        
    # --- Loop through all agents and their configs ---
    for agent_name, details in EXPERIMENTS.items():
        configs = load_config(details['config_path'])
        
        for config_name, hyperparams in configs.items():
            print(f"\n--- Starting Training for {agent_name} with config: {config_name} ---")
            
            log_dir = f"runs/{agent_name}/{config_name}"
            
            # Model path needs different extension for torch models
            model_ext = ".pt" if agent_name == "ActorCritic" else ".pkl"
            model_path = f"models/{agent_name}_{config_name}_policy{model_ext}"
            
            train(details['class'], env, hyperparams, NUM_EPISODES, log_dir, model_path)