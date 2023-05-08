import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np

from agent import Agent
from model import Model

# hyperparameters
learning_rate = 0.3
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
is_slippery = True

# enviroments
env2 = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery, render_mode="human").env
#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,render_mode="rgb_array").env
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery,render_mode="ansi").env
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

model = Model(env)
model.train_model()
model.wypisz_ruchy(4)
model.play_game_without_learning(2)




