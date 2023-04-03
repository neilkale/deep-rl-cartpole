# This is just for playing with some parameters

import gym
env = gym.make("MountainCar-v0")
observation = env.reset(seed=42)
#print(observation)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated = env.step(action)
    print(observation, reward, terminated, truncated)

    if terminated or truncated:
        observation = env.reset()
env.close()