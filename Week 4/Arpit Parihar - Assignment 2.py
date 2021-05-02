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
# **Construct the transition probability from $(s,e)$ to $(s′,e′)$ for each action $a \in \{0,1,2\}$.**
# %%
s = [0, 1, 2, 3, 4]
e = [0, 1, 2]
a = [0, 1, 2]

p_e = np.array(
    [0.8, 0.1, 0.1],
    [0.01, 0.98, 0.01],
    [0.1, 0.1, 0.8],
    dtype=float
)

P = np.zeros(len(s)*len(e), len(s)*len(e), len(a))

for i in a:
    for j in s:
        for k in e:
            