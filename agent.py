from collections import defaultdict
from gymnasium import Env

import numpy as np
import gymnasium as gym

class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        
        self.q_values = np.zeros((16,4))
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Updates the Q-value of an action."""

        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # temporal_difference = (
        #     reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        # )

        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )
        # self.training_error.append(temporal_difference)

        self.q_values[obs][action] = (1 - self.lr)*self.q_values[obs][action] + self.lr * (reward + self.discount_factor* np.max(self.q_values[next_obs])) 
        

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)