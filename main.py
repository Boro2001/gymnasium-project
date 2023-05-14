from pettingzoo.sisl import pursuit_v4

pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)

env = pursuit_v4.env(render_mode="human")
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(reward)
    print(observation)
    print(info)
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
env.close()