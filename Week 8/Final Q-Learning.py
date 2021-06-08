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
#### Model 3 - Q-Learning
# ****
# %% [markdown]
#### Importing modules
# %%

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TLG_env import garden_env

# %% [markdown]
#### Declare environment constants and transition matrices
# %%
# Dictionaries to keep track of weather, bunny and actions
WEATHER = {0: "Sunny", 1: "Rainy", 2: "Stormy"}
BUNNY = {0: "No_Bunny", 1: "Bunny"}
ACTIONS = {-1: "Pump", 0: "Rest", 1: "Water"}

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
# %% [markdown]
#### Declaring algorithm constants
# %%
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 20000
DAYS_TILL_HARVEST = 50
HARVEST_REWARD = 50

epsilon = 0.1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# %% [markdown]
#### Algorithm loop
# %%
try:
    q_table = joblib.load("Q_Table.pkl")
except:
    # Random initialization
    q_table = np.random.uniform(low=-0.5, high=0.5, size=(len(env.P_ind), len(env.A)))

    for episode in range(EPISODES):
        # Reset starting state with each episode
        env.reset_env()
        state = env.state
        for day in range(DAYS_TILL_HARVEST):

            # Explore the environment by picking random action
            if np.random.random() > epsilon:
                action = np.argmax(q_table[env.P_ind.index(state)])
            else:
                action = np.random.randint(0, len(env.A))

            # Get new state and reward from the environment
            env.step(action)
            state_new = env.state

            reward = (
                env.reward
                if day < DAYS_TILL_HARVEST - 1
                else env.reward + HARVEST_REWARD
            )

            # Calculate future Q and current Q from Q table, and new Q from modified Bellman's equation
            max_future_q = np.max(q_table[env.P_ind.index(state_new)])
            current_q = q_table[env.P_ind.index(state), action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )

            # Update Q table with new Q
            q_table[env.P_ind.index(state), action] = new_q
            state = state_new

            # End episode if terminal saturation states are reached
            if eval(state)[0] in [min(env.S), max(env.S)]:
                break
        # Adjust epsilon
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    # Save Q-table
    joblib.dump(q_table, "Q_Table.pkl")
# %% [markdown]
#### Displaying final Q-table and policy
# %%
# policy = pd.Series([x for x in q_table.argmax(axis=1) - 1], index=QL_Agent.P_ind)
Q_table_final = pd.DataFrame(q_table, columns=(env.A.values()), index=env.P_ind)

# Obtain policy from Q-table
policy = pd.Series([x for x in q_table.argmax(axis=1)], index=env.P_ind)
Q_table_final["Policy_QLearning"] = pd.Series(
    [env.A[x] for x in q_table.argmax(axis=1) - 1], index=env.P_ind
)
Q_table_final.loc[[eval(x)[0] not in [min(env.S), max(env.S)] for x in env.P_ind], :]

# %% [markdown]
#### Policy evaluation
# %%
