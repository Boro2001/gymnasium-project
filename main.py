import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np

from agent import Agent

# hyperparameters
learning_rate = 0.3
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
is_slippery=True

env2 = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery, render_mode="human").env
#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,render_mode="rgb_array").env
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery,render_mode="ansi").env
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

agent = Agent(
    env,
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon
)

observation, info = env.reset()
#
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    clear_output()
    print(episode)
    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)
        
        done = terminated or truncated
        obs = next_obs

    
    agent.decay_epsilon()


print(agent.q_values)

def wypisz_ruchy(n):
    
    for i in range(n):
        tab = []
        for j in range(n):
            index = i*4 + j
            w = np.argmax(agent.q_values[index])
            if w == 0: # lewo
                tab.append('A')
            if w == 1: # dół
                tab.append('S')
            if w == 2: # prawo
                tab.append('D')
            if w == 3: # góra
                tab.append('W')

        print(tab)

wypisz_ruchy(4)
agent.env = env2
for episode in range(10):
    obs, info = env2.reset()
    done = False
    clear_output()
    
    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = agent.env.step(action)

        #agent.update(obs, action, reward, terminated, next_obs)
        
        done = terminated or truncated
        obs = next_obs

    

# rolling_length = 500
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
# axs[0].set_title("Episode rewards")
# reward_moving_average = (
#     np.convolve(
#         np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
# axs[1].set_title("Episode lengths")
# length_moving_average = (
#     np.convolve(
#         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#     )
#     / rolling_length
# )
# axs[1].plot(range(len(length_moving_average)), length_moving_average)
# axs[2].set_title("Training Error")
# training_error_moving_average = (
#     np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
#     / rolling_length
# )
# axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
# plt.tight_layout()
# plt.show()

env.close()
env2.close()


