# %% [markdown]
# # Reinforcement Learning
# ## Final Project - The Lazy Gardener
# ### Contributors:
# - Nina Randorf
# - Remy Zhong
# - Arpit Parihar
# %% [markdown]
# ### Importing modules
# %%
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
import random
import numpy as np
import pandas as pd
from TLG_env import garden_env
# %% [markdown]
# ### Declare environment constants and transition matrices
# %%
# Dictionaries to keep track of weather, bunny and actions
WEATHER = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
BUNNY = {0: 'No_Bunny', 1: 'Bunny'}
ACTIONS = {-1: 'Pump', 0: 'Nothing', 1: 'Water'}

# List of possible saturation states
STATES = list(range(-1, 6))

# Weather to weather transition matrix
P_WEATHER = pd.DataFrame(
    {
        0: [0.75, 0.3, 0.4],
        1: [0.2, 0.5, 0.4],
        2: [0.05, 0.2, 0.2]
    })

# Weather to bunny transition matrix
P_BUNNY = pd.DataFrame(
    {
        0: [0.2, 0.8],
        1: [0.6, 0.4],
        2: [0.9, 0.1]
    })
# %% [markdown]
# ## Creating environment instance for Q-Learning algorithm
# %%
env = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
# %% [markdown]
# ### Declaring algorithm constants
# %%
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 10000
DAYS_TILL_HARVEST = 50
HARVEST_REWARD = 0

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# %% [markdown]
# ### Algorithm loop
# %%

q_table = np.random.uniform(
    low=-0.5, high=0.5, size=(len(env.P_ind), len(env.A)))

for episode in range(EPISODES):
    env.reset_env()
    state = env.state
    for day in range(DAYS_TILL_HARVEST):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[env.P_ind.index(state)])
        else:
            action = np.random.randint(0, len(env.A))

        # rand_num = np.random.random()
        # P_cur = pd.DataFrame(P[action], index=P_ind, columns=P_ind)
        # state_new = pd.Series(P_ind)[P_cur.loc[state, :].cumsum().ge(rand_num).tolist()].iloc[0]
        env.step(action)
        state_new = env.state

        # reward = R[action, P_ind.index(state), P_ind.index(state_new)]
        reward = env.reward if day < DAYS_TILL_HARVEST - \
            1 else env.reward + HARVEST_REWARD

        max_future_q = np.max(q_table[env.P_ind.index(state_new)])
        current_q = q_table[env.P_ind.index(state), action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
            (reward + DISCOUNT * max_future_q)

        q_table[env.P_ind.index(state), action] = new_q
        state = state_new
        if eval(state)[0] in [min(env.S), max(env.S)]:
            break
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
# %%
policy = pd.Series([env.A[x]
                    for x in q_table.argmax(axis=1) - 1], index=env.P_ind)
# policy = pd.Series([x for x in q_table.argmax(axis=1) - 1], index=QL_Agent.P_ind)
Q_table_final = pd.DataFrame(
    q_table, columns=(env.A.values()), index=env.P_ind)
Q_table_final['Policy_QLearning'] = policy
# Q_table_final
# # %%
# # %%
# # %%
# QL_Agent.step(-1)
# QL_Agent.state
# QL_Agent.reward

# %%
# Creating q vector
q = np.zeros((len(ACTIONS), len(env.P_ind)), dtype=np.float64)

for i in ACTIONS:
    for j in range(len(env.P_ind)):
        q[i, j] = np.dot(env.P[i, j, :], env.R[i, :])

T = 600
v = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)
d = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)

for t in range(1, T + 1):
    for i in range(len(env.P_ind)):
        rhs = np.zeros(len(ACTIONS), dtype=np.float64)
        for j in ACTIONS:
            rhs[j] = q[j, i] + DISCOUNT * \
                np.matmul(env.P[j, i, :], v[:, t - 1])
        v[i, t] = max(rhs)
        d[i, t] = np.argmax(rhs)

v = pd.DataFrame(v.T, columns=env.P_ind)
d = pd.DataFrame(d.T, columns=env.P_ind)
# %%
Q_table_final['Policy_QLearning'] = policy
policy = pd.Series([env.A[x]
                    for x in d.iloc[len(v) - 1, :] - 1], index=env.P_ind)
Q_table_final['Policy_VIter'] = policy
Q_table_final.loc[[eval(x)[0] not in [min(env.S), max(env.S)]
                   for x in env.P_ind], :]
# %%

v = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)
d = np.zeros((len(env.P_ind), T + 1), dtype=np.float64)

for t in range(1, T + 1):
    for i in range(len(env.P_ind)):
        rhs = np.zeros(len(ACTIONS), dtype=np.float64)
        for j in ACTIONS:
            rhs[j] = q[j, i] + DISCOUNT * \
                np.matmul(env.P[j, i, :], v[:, t - 1])
        v[i, t] = max(rhs)
        d[i, t] = np.argmax(rhs)

    PP = np.array([env.P[j, i, :]
                   for (i, j) in enumerate(d[:, t].astype('int'))])
    A = np.concatenate((np.identity(len(env.P_ind)) - PP,
                        np.ones((len(env.P_ind), 1))), axis=1)
    A = np.identity(len(env.P_ind)) - PP
    A = np.delete(A, len(env.P_ind) - 1, 1)

    qq = np.array([q[j, i] for (i, j) in enumerate(d[:, t].astype('int'))])
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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, day):
        self.memory.append((state, action, reward, next_state, day))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, day in minibatch:
            target = reward
            if day < (DAYS_TILL_HARVEST - 1):
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# %%
EPISODES = 2
DAYS_TILL_HARVEST = 50
HARVEST_REWARD = 50
env = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
state_size = len(eval(env.state))
action_size = len(env.A)
agent = DQNAgent(state_size, action_size)
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
        reward = env.reward if day < DAYS_TILL_HARVEST - \
            1 else env.reward + HARVEST_REWARD
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, day)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if next_state[0][0] in [min(env.S), max(env.S)]:
            break

# %%
policy_DQN = [np.argmax(agent.model.predict(np.reshape(np.array(eval(x)), [1, state_size]))) - 1 for x in env.P_ind]