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
import numpy as np
import pandas as pd
from TLG_env import garden_env
# %% [markdown]
# ### Declare environment constants and transition matrices
# %%
# Dictionaries to keep track of weather, bunny and actions
WEATHER = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
BUNNY = {0: 'No_Bunny', 1: 'Bunny'}
ACTIONS = {-1: 'Sand', 0: 'Nothing', 1: 'Water'}

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
QL_Agent = garden_env(WEATHER, BUNNY, ACTIONS, STATES, P_WEATHER, P_BUNNY)
# %% [markdown]
# ### Declaring algorithm constants
# %%
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 10000
DAYS_TILL_HARVEST = 50

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# %% [markdown]
# ### Algorithm loop
# %%

q_table = np.random.uniform(low=-10, high=0.5, size=(len(QL_Agent.P_ind), len(QL_Agent.A)))

for episode in range(EPISODES):
    QL_Agent.reset_env()
    state = QL_Agent.state
    for day in range(DAYS_TILL_HARVEST):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[QL_Agent.P_ind.index(state)])
        else:
            action = np.random.randint(0, len(QL_Agent.A))
        
        # rand_num = np.random.random()
        # P_cur = pd.DataFrame(P[action], index=P_ind, columns=P_ind)
        # state_new = pd.Series(P_ind)[P_cur.loc[state, :].cumsum().ge(rand_num).tolist()].iloc[0]
        QL_Agent.step(action)
        state_new = QL_Agent.state
        
        # reward = R[action, P_ind.index(state), P_ind.index(state_new)]
        reward = QL_Agent.reward if day < DAYS_TILL_HARVEST - 1 else QL_Agent.reward + 50
        
        max_future_q = np.max(q_table[QL_Agent.P_ind.index(state_new)])
        current_q = q_table[QL_Agent.P_ind.index(state), action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[QL_Agent.P_ind.index(state), action] = new_q
        state = state_new
        if eval(state)[0] in [min(QL_Agent.S), max(QL_Agent.S)]:
            break
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
# %%
policy = pd.Series([QL_Agent.A[x] for x in q_table.argmax(axis=1) - 1], index=QL_Agent.P_ind)
Q_table_final = pd.DataFrame(q_table, columns=(QL_Agent.A.values()), index=QL_Agent.P_ind)
Q_table_final['Policy'] = policy
Q_table_final
# # %%
# # %%
# # %%
# QL_Agent.step(-1)
# QL_Agent.state
# QL_Agent.reward

# %%
