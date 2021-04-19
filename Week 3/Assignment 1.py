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
# 1\) Plot the n-step ahead probabilities, $P_{i,j}(n)$ for $n = 1,2,...,100$ for every $i$ and $j$ in $\{AAA, AA, A, BBB, BB, B, CCC, D\}$. Make sure to properly label each state based on the rating labels.
# %%
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
# from hmmviz import TransGraph
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
%pylab inline
%config InlineBackend.figure_formats = ['png']
# %%
# Creating transition probability matrix
P = np.array([[90.81, 8.33, 0.68, 0.06, 0.08, 0.02, 0.01, 0.01],
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
# %%
# plt.figure(figsize=(15, 15))
# TransGraph(P).draw(edgelabels=True, edgewidths=1)
# %%
