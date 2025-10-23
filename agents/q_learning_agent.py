import numpy as np
from collections import defaultdict
import pickle

class QLearningAgent:
    """
    A Q-Learning agent for the 'Dagdag o Lapad' game.

    This agent learns the value of state-action pairs (Q-values) using the
    Temporal-Difference (TD) learning method. It updates its Q-table after
    every step.
    """
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initializes the agent.
        """
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-Learning formula.
        This is the core of the algorithm.
        """
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # The Q-Learning update rule
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def save_policy(self, file_path):
        """Saves the learned Q-table to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_policy(self, file_path):
        """Loads a Q-table from a file."""
        with open(file_path, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), pickle.load(f))