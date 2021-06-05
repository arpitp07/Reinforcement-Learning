# %%
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
# %% [markdown]
# ### Declare environment constants and transition matrices
# %%
# Dictionaries to keep track of weather, bunny and actions
W = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
B = {0: 'No_Bunny', 1: 'Bunny'}
A = {-1: 'Sand', 0: 'Nothing', 1: 'Water'}

# List of possible saturation states
S = list(range(-1, 6))

# Weather to weather transition matrix
P_weather = pd.DataFrame(
    {
        0: [0.75, 0.3, 0.4],
        1: [0.2, 0.5, 0.4],
        2: [0.05, 0.2, 0.2]
    })

# Weather to bunny transition matrix
P_bunny = pd.DataFrame(
    {
        0: [0.2, 0.8],
        1: [0.6, 0.4],
        2: [0.9, 0.1]
    })
# %% [markdown]
# ### Combine bunny and weather matrices
# %%
# Creating index (weather, bunny) and initializing bunny/weather transition matrix
ind = [f'({x}, {y})' for y in P_bunny.index for x in P_weather.index]
P_bw = pd.DataFrame(columns=ind, index=ind)

# Populating the bunny/weather matrix
for row in P_bw.index:
    for col in P_bw.columns:
        i = eval(row)
        j = eval(col)
        P_bw.loc[row, col] = P_weather.loc[i[0], j[0]] * \
            P_bunny.loc[j[1], j[0]]
# %% [markdown]
# ### Creating current state to future state matrix based on possible transitions:
# $$s' = s - 1 + w' + a$$
# %%
# Initializing a tensor with dimensions len(A) * len(S) * len(S)
P_s = np.zeros((len(A), len(S), len(S)), dtype=float)

# Calculating possible states based on the expression for s'
for i in A:
    for j in S:
        for k in W:
            s_new = j - 1 + k + i
            try:
                P_s[list(A).index(i), S.index(j), S.index(s_new)] = 1
            except:
                continue
# %% [markdown]
# ### Combining state, weather and bunny matrices into final transition matrix
# %%
P_ind = [f'({x}, {eval(y)[0]}, {eval(y)[1]})' for y in ind for x in S]

P_df = pd.DataFrame(
    np.zeros(
        (
            len(S) * len(W) * len(B),
            len(S) * len(W) * len(B)),
        dtype=float),
    index=P_ind, columns=P_ind)
P_list = [P_df.copy() for x in range(len(A))]

# %%

for i in range(len(P_list)):
    for j in P_df.index:
        for k in P_df.columns:
            row = eval(j)
            col = eval(k)
            if np.clip(row[0] - 1 + col[1] + list(A)[i], min(S), max(S)) == col[0]:
                P_list[i].loc[j, k] = P_s[i, S.index(row[0]), S.index(col[0])] * \
                    P_bw.loc[f'({row[1]}, {row[2]})', f'({col[1]}, {col[2]})']
    
    # Terminal state conditions - plant dies at saturation -1 and 5
    term_dry = [eval(x)[0] == min(S) for x in P_ind]
    term_wet = [eval(x)[0] == max(S) for x in P_ind]
    P_list[i].iloc[term_dry, :] = [1] + [0] * (len(term_dry) - 1)
    P_list[i].iloc[term_wet, :] = [0] * (len(term_wet) - 1) + [1]

# %%
P = np.stack((P_list[0], P_list[1], P_list[2]))
# %%
R = np.zeros_like(P)

for i in range(len(A)):
    for j in range(len(P_ind)):
        for k in range(len(P_ind)):
            curr = eval(P_ind[j])
            next = eval(P_ind[k])
            R[i, j, k] = (-(curr[0] - 1)**2 + 2 -
                          (10 * curr[2]) - abs(list(A)[i])) * 0.1
    term_dry = [eval(x)[0] == min(S) for x in P_ind]
    term_wet = [eval(x)[0] == max(S) for x in P_ind]
    R[i, term_dry, :] = [-100] * len(term_dry)
    R[i, term_wet, :] = [-100] * len(term_wet)
# %%
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 2000
DAYS_TILL_HARVEST = 50

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-10, high=0.5, size=(len(P_ind), len(A)))

for episode in range(EPISODES):
    state = np.random.choice(P_ind)
    for day in range(DAYS_TILL_HARVEST):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[P_ind.index(state)])
        else:
            action = np.random.randint(0, len(A))
        
        rand_num = np.random.random()
        P_cur = pd.DataFrame(P[action], index=P_ind, columns=P_ind)
        state_new = pd.Series(P_ind)[P_cur.loc[state, :].cumsum().ge(rand_num).tolist()].iloc[0]
        
        reward = R[action, P_ind.index(state), P_ind.index(state_new)]
        
        max_future_q = np.max(q_table[P_ind.index(state_new)])
        current_q = q_table[P_ind.index(state), action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[P_ind.index(state), action] = new_q
        state = state_new
        if eval(state)[0] in [min(S), max(S)]:
            break
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
# %%
policy = pd.Series(q_table.argmax(axis=1) - 1, index=P_ind)
# %%
class garden_env():
    def __init__(self, W, B, A, S, P_weather, P_bunny):
        self.W = W
        self.B = B
        self.A = A
        self.S = S
        self.P_weather = P_weather
        self.P_bunny = P_bunny
        
    def create_P_bw(self):
        # Creating index (weather, bunny) and initializing bunny/weather transition matrix
        ind = [f'({x}, {y})' for y in self.P_bunny.index for x in self.P_weather.index]
        self.P_bw = pd.DataFrame(columns=ind, index=ind)

        # Populating the bunny/weather matrix
        for row in self.P_bw.index:
            for col in self.P_bw.columns:
                i = eval(row)
                j = eval(col)
                self.P_bw.loc[row, col] = self.P_weather.loc[i[0], j[0]] * \
                    self.P_bunny.loc[j[1], j[0]]
        
        return self.P_bw