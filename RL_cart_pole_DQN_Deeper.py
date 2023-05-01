import numpy as np
import pfrl
import torch
import torch.nn
import gym
import numpy
import matplotlib.pyplot as plt
import addcopyfighandler
#from gym.wrappers.record_video import RecordVideo
from torchviz import make_dot
from torchsummary import summary

env = gym.make('CartPole-v1')
print(env)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
print(action)
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)

# Uncomment to open a GUI window rendering the current state of the environment
env.render()

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 160)
        self.l2 = torch.nn.Linear(160, 320)
        self.l3 = torch.nn.Linear(320, 640)
        self.l4 = torch.nn.Linear(640, 1280)
        self.l5 = torch.nn.Linear(1280, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = torch.nn.functional.relu(self.l3(h))
        h = torch.nn.functional.relu(self.l4(h))
        h = self.l5(h)
        return pfrl.action_value.DiscreteActionValue(h)

obs_size = env.observation_space.low.size
print(obs_size)
n_actions = env.action_space.n
print(env.action_space, n_actions)
q_func = QFunction(obs_size, n_actions)
print(q_func)

print(summary(q_func, input_size=(1, 4)))

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=0.01)
print(optimizer)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
print(explorer)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
print(replay_buffer)

x = torch.randn(1, 4)
y = q_func(x)
print(y.q_values)
print(dir(y))
print(dir(q_func))
print(q_func.__getattr__)
print(type(y))
print(x, y)
print(y.q_values.mean())
make_dot(y.q_values.mean(), params=dict(q_func.named_parameters()),
         show_attrs=False, show_saved=False).render("ddqn_torchviz", format="png")



# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(numpy.float32, copy=False)
print(phi)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1



# Now create an agent that will interact with the environment.
agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)
print(agent)

reward_val = []
reward_avg = []
reward_max = []
n_episodes = 200
max_episode_len = 100
for i in range(1, n_episodes + 1):
    obs = env.reset()
    env.render()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    reward_avg_list = []
    while True:
        # Uncomment to watch the behavior in a GUI window
        env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        reward_val.append(R)
        reward_avg_list.append(R)
        if done or reset:
            break
        if i % 10 == 0:
            print('episode:', i, 'R:', R, 't:', t)
        if i % 50 == 0:
            print('statistics:', agent.get_statistics())
    reward_avg.append(sum(reward_avg_list)/len(reward_avg_list))
    reward_max.append(max(reward_avg_list))
    print(i, 'Finished.')

plt.plot(reward_max)
plt.xlabel('Episodes')
plt.ylabel('Reward')
reward_max = np.asarray(reward_max)
plt.title(str(max_episode_len) + ' per episode for ' + str(n_episodes) + ' episodes during training (Average: ' +
          str(round(np.mean(reward_max), 1)) + u"\u00B1" + str(round(np.std(reward_max), 1)) + ')')
plt.show()

reward_values_eval = []

with agent.eval_mode():
    for i in range(500):
        obs = env.reset()
        env.render()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)
        reward_values_eval.append(R)

print(reward_values_eval)
reward_values_eval = np.asarray(reward_values_eval)
print(np.mean(reward_values_eval), np.std(reward_values_eval))
plt.plot(reward_values_eval)
plt.title('Reward over episodes during testing (Average: ' + str(round(np.mean(reward_values_eval), 1)) + u"\u00B1" + str(round(np.std(reward_values_eval), 1)) + ')')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()

# Save an agent to the 'agent' directory
agent.save('agent')

# Uncomment to load an agent from the 'agent' directory
# agent.load('agent')

# # Set up the logger to print info messages for understandability.
# import logging
# import sys
# logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
#
# pfrl.experiments.train_agent_with_evaluation(
#     agent,
#     env,
#     steps=2000,           # Train the agent for 2000 steps
#     eval_n_steps=None,       # We evaluate for episodes, not time
#     eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
#     train_max_episode_len=200,  # Maximum length of each episode
#     eval_interval=1000,   # Evaluate the agent after every 1000 steps
#     outdir='result',      # Save everything to 'result' directory
# )
#
