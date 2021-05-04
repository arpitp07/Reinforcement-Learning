# %% [markdown]
# # Reinforcement Learning - Assignment 2
# ## Arpit Parihar
# ## 04/19/2021
# ****
# %% [markdown]
# A child has available a certain number of ice cream scoops every day, $s$. The child can store a number of scoops for the next day $a$ and eat the remainder scoops $c = s − a$. Mathematically,
# $$s = a + c$$
# where $a \in \{0, 1, .., s\}$. The number of scoops available the next day, $s'$, is equal to the number of scoops stored over night, $a$, plus an additional scoops provided by the parents, $e$. The number of scoops available the next day is given by:
#
# $$
# \begin{eqnarray}
# s'= && a + e' \\
#   = && (s - c) + e'
# \end{eqnarray}
# $$
#
# where $e' \in \{0, 1, 2\}$. $e'$ is known after action $a$ is taken. The child can store up to $2$ scoops in the fridge every day which implies that $a \in \{0, 1, 2\}$ and $s \in \{0, 1, 2, 3, 4\}$. The transition probability matrix from $e$ to $e'$ is given by:
#
# $$P = \begin{bmatrix} 0.8 & 0.1 & 0.1 \\ 0.01 & 0.98 & 0.01 \\ 0.1 & 0.1 & 0.8 \end{bmatrix}$$
# %% [markdown]
# Importing modules
# %%
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import joblib
import math
import gym
from gym import spaces
import random
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')
# %% [markdown]
# - **Construct the transition probability from $(s,e)$ to $(s′,e′)$ for each action $a \in \{0,1,2\}$.**
# %%
s = range(5)
e = range(3)
a = range(3)

p_e = np.array([
    [0.8, 0.1, 0.1],
    [0.01, 0.98, 0.01],
    [0.1, 0.1, 0.8]],
    dtype=float
)

P = np.zeros((len(a), len(s) * len(e), len(s) * len(e)), dtype=np.float64)
R = np.zeros((len(a), len(s) * len(e), len(s) * len(e)), dtype=np.float64)

for i in a:
    for j in range(len(s) * len(e)):
        P[i, j, i + np.arange(len(e)) * (len(s) + 1)] = p_e[j // len(s)]
        R[i, j, :] = math.log(
            j % len(s) - i + 1) if j % len(s) - i >= 0 else -math.inf

states = [f'({y}, {x})' for x in e for y in s]
P_mat = [pd.DataFrame(P[x, :, :], columns=states, index=states) for x in a]
for x in range(len(P_mat)):
    print(f'Transition matrix for a = {x}:\n')
    P_mat[x]
    print('\n')
# %% [markdown]
# - **The child subjected utility from eating ice cream is $log(c+1)$ if $c≥0$ and $−\infty$ otherwise. Construct the reward matrix for each transition $(s,e,a)$ and $(s′,e′)$.**
# %%
R_mat = [pd.DataFrame(R[x, :, :], columns=states, index=states) for x in a]
for x in range(len(R_mat)):
    print(f'Reward matrix for a = {x}:\n')
    R_mat[x]
    print('\n')
# %% [markdown]
# - **Apply the value iteration approach to compute the value function for each state $(s,e)$ and optimal policy. Iterate $600$ times.**
# %%
# Creating q vector
q = np.zeros((len(a), len(s) * len(e)), dtype=np.float64)

for i in a:
    for j in range(len(s) * len(e)):
        q[i, j] = np.nan_to_num(np.dot(P[i, j, :], R[i, j, :]), nan=-math.inf)

T = 600
v = np.zeros((len(states), T + 1),dtype=np.float64)
d = np.zeros((len(states), T + 1),dtype=np.float64)

for t in range(1, T + 1):
    for i in range(len(states)):
        rhs = np.zeros(len(a), dtype=np.float64)
        for j in a:
            rhs[j] = q[j, i] + np.matmul(P[j, i, :], v[:, t - 1])
        v[i, t] = max(rhs)
        d[i, t] = np.argmax(rhs)

v = pd.DataFrame(v.T, columns=states)
d = pd.DataFrame(d.T, columns=states)
# %% [markdown]
# - **For each $e$, plot the value function $v(s,e)$ with $s$ on the x-axis. (hint: there should be 3 lines plots, one for each $e\in\{0,1,2\}$).**
# %%
plot_data = np.zeros((len(s), len(e)))
for i in range(len(states)):
    row = eval(v.iloc[-1].index[i])[0]
    column = eval(v.iloc[-1].index[i])[1]
    plot_data[row, column] = v.iloc[-1, i]

plt.plot(plot_data)
plt.xlabel('s')
plt.xticks(ticks=range(5))
plt.ylabel('Expected Value')
plt.legend(e)
plt.title('Average Value for each s and e')
plt.show();