import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    # --- CHANGE: The state_size is now 4 ---
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(64, action_size)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return torch.softmax(action_logits, dim=-1), state_value

class ActorCriticAgent:
    # --- CHANGE: The default state_size is now 4 ---
    def __init__(self, action_space, state_size=4, actor_lr=0.001, critic_lr=0.005, discount_factor=0.99):
        self.action_space = action_space
        self.gamma = discount_factor
        
        self.policy_network = ActorCriticNetwork(state_size, len(action_space))
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        
        self.log_probs = []
        self.rewards = []
        self.values = []

    def _format_state(self, state):
        return torch.FloatTensor(state).unsqueeze(0)

    def choose_action(self, state):
        state_tensor = self._format_state(state)
        action_probs, state_value = self.policy_network(state_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        
        return action.item()

    def learn(self, reward, done):
        self.rewards.append(reward)
        if not done:
            return

        R = 0
        policy_loss = []
        value_loss = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.smooth_l1_loss(value.squeeze(0), torch.tensor([R])))

        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self.values = []

    def save_policy(self, file_path):
        torch.save(self.policy_network.state_dict(), file_path)

    def load_policy(self, file_path):
        self.policy_network.load_state_dict(torch.load(file_path))