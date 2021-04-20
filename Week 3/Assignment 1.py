# %% [markdown]
# # Reinforcement Learning - Assignment 1
# ## Arpit Parihar
# ## 04/19/2021
# ****
# %% [markdown]
# ## Part 1 - Bond rating transitions
# Consider the following transition probability matrix for corporate bond credit ratings:
# $$P = \begin{bmatrix} 90.81 & 8.33 & 0.68 & 0.06 & 0.08 & 0.02 & 0.01 & 0.01 \\ 0.70 & 90.65 & 7.79 & 0.64 & 0.06 & 0.13 & 0.02 & 0.01 \\ 0.09 & 2.27 & 91.05 & 5.52 & 0.74 & 0.26 & 0.01 & 0.06 \\0.02 & 0.33 & 5.95 & 85.93 & 5.30 & 1.17 & 1.12 & 0.18 \\0.03 & 0.14 & 0.67 & 7.73 & 80.53 & 8.84 & 1.00 & 1.06 \\0.01 & 0.11 & 0.24 & 0.43 & 6.48 & 83.46 & 4.07 & 5.20 \\0.21 & 0.00 & 0.22 & 1.30 & 2.38 & 11.24 & 64.86 & 19.79 \\0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 100.00 \end{bmatrix}$$
# The ratings are AAA, AA, A, BBB, BB, B, CCC, D and represent states 1-8 in the Markov chain.
# %% [markdown]
# **1\) Plot the n-step ahead probabilities, $P_{i,j}(n)$ for $n = 1,2,...,100$ for every $i$ and $j$ in $\{AAA, AA, A, BBB, BB, B, CCC, D\}$. Make sure to properly label each state based on the rating labels.**
# %%
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')
%config InlineBackend.figure_formats = ['png']
# %%
# Creating transition probability matrix
P = np.array([
    [90.81, 8.33, 0.68, 0.06, 0.08, 0.02, 0.01, 0.01],
    [0.70, 90.65, 7.79, 0.64, 0.06, 0.13, 0.02, 0.01],
    [0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06],
    [0.02, 0.33, 5.95, 85.93, 5.30, 1.17, 1.12, 0.18],
    [0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06],
    [0.01, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20],
    [0.21, 0.00, 0.22, 1.30, 2.38, 11.24, 64.86, 19.79],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00]],
    dtype=float)
P = P / np.sum(P, axis=1)
labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
P = pd.DataFrame(P, index=labels, columns=labels)
P
# %%
T = 100
n = len(P)
P_cumulative = np.zeros((T, n, n), dtype=np.float64)
P_cumulative[0] = P
for t in range(T - 1):
    P_cumulative[t + 1] = np.matmul(P_cumulative[t], P)
# %%
fig, axs = plt.subplots(len(P), len(P), figsize=(50, 50));
for i in range(len(P)):
    for j in range(len(P)):
        axs[i, j].plot(P_cumulative[:, i, j]);
        axs[i, j].set_title(f'$P_{{{labels[i]}, {labels[j]}}}(n)$')
        axs[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i, j].set(xlabel='n', ylabel='Probability')
        axs[i, j].grid(linestyle='dashed')
# %% [markdown]
# **2\) Group all states into classes. Report the classes and the total number of classes. Explain your answer.**
#
# The states can be grouped into 2 classes:
# - $\{AAA, AA, A, BBB, BB, B, CCC\}$
# - $\{D\}$
#
# All the states communicate with each other except for D, which is an absorbing state. State $CCC$ doesn't communicate with state $AA$ in a single step, but does so in multiple steps.
# %% [markdown]
# **3\) What is the periodicity of the transition probability matrix? Explain your answer.**
#
# All the states have access to themselves in a single step, so the periodicity of the transition matrix is $d = 1$. In other words, the chain is aperiodic.
# %% [markdown]
# **4) Construct a 100-step simulation for a bond rating assuming an initial $AAA$ rating and the above transition matrix.**
# %%
# simulate from uniform[0,1]
def MC_sim(K, S, p, P, seed=0):
    np.random.seed(seed)
    U = np.random.uniform(low=0.0, high=1.0, size=K)
    Y = np.zeros(K, dtype=int)
    p = pd.Series(p, index=P.index)
    P_sim = pd.DataFrame(np.zeros_like(P), index=P.index, columns=P.columns)
    k = 0
    for u in U:
        # simulation of initial values
        if k == 0:
            i = S[p.cumsum().ge(u)][0]
            j = S[p.cumsum().ge(u)][0]
            Y[k] = i
        # simulation of the transitions
        else:
            j = S[P.cumsum(axis=1).iloc[i - 1, :].ge(u)][0]
            Y[k] = j
        k += 1
        P_sim.iloc[i-1, j-1]+=1
        i = j
    return Y, P_sim

K = 100
S = pd.Series(range(1, 9), index=P.index)
p = [1, 0, 0, 0, 0, 0, 0, 0]
Y, _ = MC_sim(K, S, p, P, 7)
# %% [markdown]
# - Plot the sequence of transitions as a step function of time.
# %%
plt.plot(Y);
plt.yticks(S.values, S.index);
plt.xlabel('Time, t');
plt.ylabel('State');
plt.title('$Y_{t}$');
# %% [markdown]
# - Report the likehood of each transition and the likelihood of the entire simulated sequence.
# %%
# Running 10000 simulations with 100 steps
try:
    P_sim_total = joblib.load('sim_probs.pkl')
except:
    P_sim_total = pd.DataFrame(np.zeros_like(P), index=P.index, columns=P.columns)
    for i in range(10000):
        _, P_sim = MC_sim(K, S, p, P, None)
        P_sim_total = P_sim_total + P_sim
    joblib.dump(P_sim_total, 'sim_probs.pkl');
P_sim_probs = np.round(P_sim_total / np.sum(P_sim_total, axis=1), 4)
# %% [markdown]
# - Is this Markov chain stationary? Please provide detailed reasoning.

# - Is this Markov chain time homogeneous? Please provide detailed reasoning.
# %% [markdown]
# **5\) Compute the expected number of transitions between any pair of transient states before transitioning to the absorbing state.**
# %% [markdown]
# **6\) Compute the probability that a state $j$ will ever be reached from state $i$ (for all $i$ and $j$)**
# %% [markdown]
# **7\) Compute the probability that a bond will reach:**
#
# - $AAA$ rating within $5$ periods given a current rating of $AAA, AA, A, BBB, BB, B, CCC$
#
# - $CCC$ rating within $5$ periods given a current rating of $AAA, AA, A, BBB, BB, B, CCC$
#
# - Use your intuition and guess whether $f_{i,i}<1$ or $f_{i,i}=1$ for each rating?
