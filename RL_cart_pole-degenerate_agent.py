import pfrl
import torch
import torch.nn
import gym
import numpy
#from gym.wrappers.record_video import RecordVideo

env = gym.make('CartPole-v1')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)

# Uncomment to open a GUI window rendering the current state of the environment
env.render()

class DegenAgent():
    def act(self):
        return 0

agent = DegenAgent()

n_episodes = 300
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        env.render()
        action = agent.act()
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
print('Finished.')

for i in range(10):
    obs = env.reset()
    R = 0
    t = 0
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act()
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
        reset = t == 200
        if done or reset:
            break
    print('evaluation episode:', i, 'R:', R)
