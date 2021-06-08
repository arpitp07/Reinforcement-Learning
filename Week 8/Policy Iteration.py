# %% [markdown]
## The Lazy Gardener: Selecting an Optimal Watering Strategy through Reinforcement Learning
#### MSCA 32020 Reinforcement Learning
##### Spring 2021
##### Team Members:
# - Nina Randorf
# - Remy Zhong
# - Arpit Parihar
##
# ****
#### Model 2 - Policy Iteration
# ****
# %% [markdown]
#### Importing modules
# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DQN import DQNAgent_1NN, DQNAgent_2NN
from TLG_env import garden_env

# %% [markdown]
#### Declare environment constants and transition matrices
# %%
# Dictionaries to keep track of weather, bunny and actions
WEATHER = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
BUNNY = {0: 'No_Bunny', 1: 'Bunny'}
ACTIONS = {-1: 'Pump', 0: 'Rest', 1: 'Water'}

# List of possible saturation states
STATES = list(range(-1, 6))

# Weather to weather transition matrix
P_WEATHER = pd.DataFrame({0: [0.75, 0.3, 0.4], 1: [0.2, 0.5, 0.4], 2: [0.05, 0.2, 0.2]})

# Weather to bunny transition matrix
P_BUNNY = pd.DataFrame({0: [0.2, 0.8], 1: [0.6, 0.4], 2: [0.9, 0.1]})
# %% [markdown]
#### Creating environment instance for Q-Learning algorithm
# %%
env = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
# %%
# Creating q vector
q = np.zeros((len(ACTIONS), len(env.P_ind)), dtype=np.float64)

for i in ACTIONS:
    for j in range(len(env.P_ind)):
        q[i, j] = np.dot(env.P[i, j, :], env.R[i, :])

DISCOUNT = 0.99
T = 600
v = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)
d = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)

for t in range(1, T + 1):
    for i in range(len(env.P_ind)):
        rhs = np.zeros(len(ACTIONS), dtype=np.float64)
        for j in ACTIONS:
            rhs[j] = q[j, i] + DISCOUNT * np.matmul(env.P[j, i, :], v[:, t - 1])
        v[i, t] = max(rhs)
        d[i, t] = np.argmax(rhs)

v = pd.DataFrame(v.T, columns=env.P_ind)
d = pd.DataFrame(d.T, columns=env.P_ind)

# %%

v = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)
d = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)

for t in range(1, T + 1):
    for i in range(len(env.P_ind)):
        rhs = np.zeros(len(ACTIONS), dtype=np.float64)
        for j in ACTIONS:
            rhs[j] = q[j, i] + DISCOUNT * np.matmul(env.P[j, i, :], v[:, t - 1])
        v[i, t] = max(rhs)
        d[i, t] = np.argmax(rhs)

    PP = np.array([env.P[j, i, :] for (i, j) in enumerate(d[:, t].astype("int"))])
    A = np.concatenate(
        (np.identity(len(env.P_ind)) - PP, np.ones((len(env.P_ind), 1))), axis=1
    )
    A = np.identity(len(env.P_ind)) - PP
    A = np.delete(A, len(env.P_ind) - 1, 1)

    qq = np.array([q[j, i] for (i, j) in enumerate(d[:, t].astype("int"))])
    tmp = np.matmul(np.linalg.inv(A), qq)
    g = tmp[len(env.P_ind) - 1]
    tmp[len(env.P_ind) - 1] = 0
    v[:, t] = tmp.T

v = pd.DataFrame(v.T, columns=env.P_ind)
d = pd.DataFrame(d.T, columns=env.P_ind)
d.iloc[-1]
# %% [markdown]
# DQN
# %%
EPISODES = 2
DAYS_TILL_HARVEST = 50
HARVEST_REWARD = 50
env = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
state_size = len(eval(env.state))
action_size = len(env.A)
agent = DQNAgent_1NN(state_size, action_size)
batch_size = 32

for e in range(EPISODES):
    env.reset_env()
    state = np.array(eval(env.state))
    state = np.reshape(state, [1, state_size])
    for day in range(DAYS_TILL_HARVEST):
        # env.render()
        action = agent.act(state)
        env.step(action)
        next_state = np.array(eval(env.state))
        reward = (
            env.reward if day < DAYS_TILL_HARVEST - 1 else env.reward + HARVEST_REWARD
        )
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, day)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if next_state[0][0] in [min(env.S), max(env.S)]:
            break

# %%
DAYS_TILL_HARVEST = 50
HARVEST_REWARD = 50
env = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
state_size = len(eval(env.state))
action_size = len(env.A)
batch_size = 32
DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
# Minimum number of steps in a memory to start training
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
# MODEL_NAME = '2x256'
# MEMORY_FRACTION = 0.20

agent = DQNAgent_2NN(state_size, action_size)

# Environment settings
EPISODES = 1000

# Exploration settings
epsilon = 0.1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Iterate over episodes
for episode in range(EPISODES):

    print(f"Starting episode {episode + 1}")
    step = 1

    # Reset environment and get initial state
    env.reset_env()
    current_state = np.array(eval(env.state))

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, action_size)

        # new_state, reward, done = env.step(action)
        env.step(action)
        new_state = np.array(eval(env.state))
        reward = env.reward if step < DAYS_TILL_HARVEST else env.reward + HARVEST_REWARD
        if new_state[0] in [min(env.S), max(env.S)]:
            print(f"Plant died on day {step}")
            done = True

        if step == DAYS_TILL_HARVEST:
            print(f"Tomatoes harvested on episode {episode + 1}")
            done = True

        # Transform new continous state to new discrete state and count reward
        # episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
# %%
label = env.S

# plt.plot(list(optimal_policy_v[::6][1:6]),
#          color='orange', label="(w,b) = (0,0)")
