import numpy as np
from collections import defaultdict
import pickle

class MonteCarloAgent:
    """
    A Monte Carlo agent for the 'Dagdag o Lapad' game.

    This agent uses a first-visit Monte Carlo method to learn the value of
    state-action pairs (Q-values). It learns only from complete episodes.
    """
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initializes the agent.

        Args:
            action_space (list): The list of possible actions.
            learning_rate (float): The learning rate (alpha).
            discount_factor (float): The discount factor (gamma).
            epsilon (float): The exploration rate for the epsilon-greedy policy.
        """
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table to store state-action values
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))
        
        # To calculate average returns
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action to take.
        """
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.action_space)
        else:
            # Exploit: choose the best known action
            return np.argmax(self.q_table[state])

    def learn(self, episode):
        """
        Updates the Q-table based on a completed episode.

        Args:
            episode (list): A list of (state, action, reward) tuples from the episode.
        """
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma**i for i in range(len(rewards) + 1)])
        
        # This is the core of Monte Carlo: we only update after the episode is done.
        for i, state in enumerate(states):
            # Calculate the total future discounted reward (the "return") from this state
            future_rewards = rewards[i:]
            G = sum(rewards[j] * discounts[j-i] for j in range(i, len(states)))
            
            # Update our Q-table using the incremental mean formula
            # This is more numerically stable than storing a list of all returns
            self.returns_sum[(state, actions[i])] += G
            self.returns_count[(state, actions[i])] += 1.0
            
            # Q(s, a) is the average of all returns G seen after visiting (s, a)
            self.q_table[state][actions[i]] = self.returns_sum[(state, actions[i])] / self.returns_count[(state, actions[i])]

    def save_policy(self, file_path):
        """Saves the learned Q-table to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_policy(self, file_path):
        """Loads a Q-table from a file."""
        with open(file_path, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), pickle.load(f))