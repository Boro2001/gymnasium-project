import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import datetime

from agent import Agent

class Model:

    # So model basically contains the agent
    # and the interface for the learning and playing games

    def __init__(self, env, n_episodes=10000, learning_rate=0.3, start_epsilon=1.0, epsilon_decay=-1,
                 final_epsilon=0.1):
        self.n_episodes = n_episodes
        self.env = env
        self.agent = Agent(self.env,
                           learning_rate,
                           start_epsilon,
                           epsilon_decay,
                           final_epsilon)
        self.obs, self.info = self.env.reset()

    def train_model(self):
        episode_steps = []  # list of number of steps per episode
        for episode in range(self.n_episodes):
            self.obs, self.info = self.env.reset()
            done = False
            clear_output()
            print(episode)
            # play one episode and get the stats about it
            number_of_steps = 0
            while not done:
                action = self.agent.get_action(self.obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(self.obs, action, reward, terminated, next_obs)
                print("is terminated " + str(terminated))
                print("is truncated " + str(truncated))
                number_of_steps = number_of_steps + 1
                print("number of steps" + str(number_of_steps))
                done = terminated or truncated
                obs = next_obs
                print(self.agent.q_values)
            self.agent.decay_epsilon()
            episode_steps.append(number_of_steps)

        plt.plot(range(self.n_episodes), episode_steps)
        plt.xlabel("Episode Number")
        plt.ylabel("Number of Steps")
        plt.title("Number of Steps per Episode")
        plt.savefig("charts/" + str(datetime.datetime.now()) + "steps_per_episode.png")
        plt.show()
        print(self.agent.q_values)

    def wypisz_ruchy(self, n):
        for i in range(n):
            tab = []
            for j in range(n):
                index = i * 4 + j
                w = np.argmax(self.agent.q_values[index])
                if w == 0:  # lewo
                    tab.append('A')
                if w == 1:  # dół
                    tab.append('S')
                if w == 2:  # prawo
                    tab.append('D')
                if w == 3:  # góra
                    tab.append('W')
            print(tab)

    def play_game_without_learning(self, n = 3):
        self.agent.env = self.env
        for episode in range(n):
            obs, info = self.env.reset()
            done = False
            clear_output()

            # play one episode
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.agent.env.step(action)
                done = terminated or truncated
                obs = next_obs

        self.env.close()

