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
%config InlineBackend.figure_formats = ['png']
%matplotlib inline
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
# Function to simulate markov chain from uniform[0,1]


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
        P_sim.iloc[i - 1, j - 1] += 1
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
# Load simulated matrix from disk if it exists, create it if it doesn't
try:
    P_sim_total = joblib.load('sim_probs.pkl')
except:
    # Running 10000 simulations with 100 steps
    P_sim_total = pd.DataFrame(np.zeros_like(
        P), index=P.index, columns=P.columns)
    for i in range(10000):
        _, P_sim = MC_sim(K, S, p, P, None)
        P_sim_total = P_sim_total + P_sim
    joblib.dump(P_sim_total, 'sim_probs.pkl');
P_sim_probs = np.round(P_sim_total / np.sum(P_sim_total, axis=1), 4)
print(f'Transition likelihoods (simulated): \n\n')
display(P_sim_probs)
print(f'\n\nTransition likelihoods (actual): \n\n')
display(P)
print('\n')
# %%
# Calculating total likelihood of the simulated sequence
ll_total = 0
for i in range(len(Y) - 1):
    ll_total += math.log(P_sim_probs.iloc[Y[i] - 1, Y[i + 1] - 1])
print(f'Total likelihood = {math.e**ll_total}\n')
# %% [markdown]
# - Is this Markov chain stationary? Please provide detailed reasoning.
#
# No, this chain is not stationary, as initial state distribution changes when multiplied with the transition matrix
# %%
print(f'p = {p}\np*P = {[round(x, 4) for x in np.matmul(p, np.array(P))]}\n')
# %% [markdown]
# Initial state distribution to make the chain a stationary chain can be obtained as below:
# %%
eig_val_1_index = np.where(np.linalg.eig(P.T)[0] == 1)[
    0][0]  # index of eigenvalue = 1
eig_vec = np.linalg.eig(P.T)[1]  # all eigenvectors
init_dist = []
for ev in eig_vec:
    init_dist.append(ev[eig_val_1_index] / np.sum(eig_vec[:, eig_val_1_index]))
print(f'Initial state distribution for stationarity:\n{init_dist}\n')
# %% [markdown]
# - Is this Markov chain time homogeneous? Please provide detailed reasoning.
#
# Yes, the Markov chain is time homogenous, as the transition matrix does not depend on time. Also, the probability of going from state $i$ at time $t$ to state $j$ at time $t+1$ remains the same ($P_{i,j}$), no matter what happened before, i.e.
# $$Prob[Y_{t+1}=j | Y_{t}=i] = P_{i,j}$$
# %% [markdown]
# **5\) Compute the expected number of transitions between any pair of transient states before transitioning to the absorbing state.**
#
# The expected number of transitions before reaching absorbing state from state $i$ to state $j$, can be obtained by solving the system of equations:
# $$Z = I_{N_{k}\times N_{k}} + H_{N_{k}\times N_{k}}Z$$
# Which is,
# $$Z = (I_{N_{k}\times N_{k}} - H_{N_{k}\times N_{k}})^{-1}$$
# %%
H = P.drop(index='D', columns='D')
Z = pd.DataFrame(np.linalg.inv(np.identity(
    H.shape[0]) - H), index=H.index, columns=H.columns)
print(f'Expected number of transitions before reaching absorbing state: \n\n')
display(Z)
# %% [markdown]
# **6\) Compute the probability that a state $j$ will ever be reached from state $i$ (for all $i$ and $j$)**
# %%
Q = pd.DataFrame(np.zeros_like(Z), index=Z.index, columns=Z.columns)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Q.iloc[i, j] = (Z.iloc[i, j] - (1 if i == j else 0)) / Z.iloc[j, j]
print(f'Probabilities of ever reaching state j from state i: \n\n')
display(Q)
# %% [markdown]
# **7\) Compute the probability that a bond will reach:**
# %%


def bond_steps(P, T, end):
    N = P.shape[0]
    f = np.zeros((T, N, N), dtype=np.float64)
    Pbar = np.zeros((N, N, N), dtype=np.float64)

    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                if k != i:
                    Pbar[i, j, k] = P.iloc[j, k]
                else:
                    Pbar[i, j, k] = 0

    for j in range(0, N):
        for t in range(0, T):
            if t == 0:
                f[t, :, j] = P.iloc[:, j]
            else:
                f[t, :, j] = np.matmul(Pbar[j, :, :], f[t - 1, :, j])

    f_final = np.zeros_like(f[0, :, 0])
    for t in range(T):
        f_final += f[t, :, end]

    return f_final


# %% [markdown]
# - $AAA$ rating within $5$ periods given a current rating of $AAA, AA, A, BBB, BB, B, CCC$
# %%
f_AAA = pd.DataFrame(bond_steps(P, 5, P.columns.get_loc(
    'AAA')), index=P.index, columns=['Probability'])
print(f'Probabilities of reaching AAA rating: \n\n{f_AAA.drop(index="D")}')
# %% [markdown]
# - $CCC$ rating within $5$ periods given a current rating of $AAA, AA, A, BBB, BB, B, CCC$
# %%
f_CCC = pd.DataFrame(bond_steps(P, 5, P.columns.get_loc(
    'CCC')), index=P.index, columns=['Probability'])
print(f'Probabilities of reaching CCC rating: \n\n{f_CCC.drop(index="D")}')
# %% [markdown]
# - Use your intuition and guess whether $f_{i,i}<1$ or $f_{i,i}=1$ for each rating?
#
# For all states except $D$, $f_{i,i}<1$, as $D$ is an absorbing state, and eventually, every other transient state will be visited for the last time. For $D$, $f_{i,i}=1$ as once $D$ is hit, the chain stays there forever.
# %% [markdown]
# ## Part 1 - Mobile robot
#
# Consider the mobile robot problem described in section 2.5 in the “Introduction” lecture. Note that the results will differ from those in the lecture since the data is simulated.
# %%
# Creating environment


class Segment:  # (gym.Env):
    def __init__(self, numActions, startObservation, p, terminal, done):
        self.numActions = numActions
        self.observation = startObservation
        self.p = p
        self.terminal = terminal
        self.action_space = spaces.Discrete(
            self.numActions)  # {0:left, 1:right}
        self.observation_space = spaces.Discrete(2 * self.terminal + 1)
        self.done = done

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(self.observation)
        assert self.action_space.n == 2
        assert self.observation != 0
        assert self.observation != (2 * self.terminal)
        observation = self.observation
        done = self.done
        if action == 0:
            observation_next = observation + \
                np.random.choice([-1, 1], p=[self.p, 1 - self.p])
        elif action == 1:
            observation_next = observation + \
                np.random.choice([-1, 1], p=[1 - self.p, self.p])
        if observation_next == (2 * self.terminal):
            done = True
        elif observation_next == 0:
            done = True
        self.done = done
        self.observation = observation_next
        reward = self.rewards(observation, observation_next, action)
        return [self.observation, reward, done]

    def rewards(self, observation, observation_next, action):
        if observation_next == (2 * self.terminal):
            reward = 1.0
        elif observation_next == 0:
            reward = -1.0
        else:
            reward = -0.05
        return reward

    def reset(self):
        observation = self.startObservation
        self.observation = observation
        self.done = False
        return observation


def policy(observation, p):
    action = 1 if random.random() <= p else 0
    return np.int(action)
# %%


def run_sim(p, o=10):
    np.random.seed = 0
    t = 0
    tMAX = 50
    done = False
    observation = np.int(o)
    observations = [observation]
    rewards = []
    actions = []

    env = Segment(
        numActions=2, startObservation=observation,
        p=0.8, terminal=10, done=False)
    while t < tMAX and done == False:
        action = policy(observation=observation, p=p)
        observation_next, reward, done = env.step(action=action)
        observations.append(observation_next)
        rewards.append(reward)
        actions.append(action)
        observation = observation_next
        t += 1

    dta = pd.DataFrame([
        range(0, t),
        observations[0:t],
        observations[1:(t + 1)],
        actions[0:t],
        rewards]).transpose()

    dta.columns = ['t', 'observation', 'observation_next', 'action', 'reward']
    dta['observation'] = dta['observation'] - 10
    dta['observation_next'] = dta['observation_next'] - 10
    dta['beta'] = 0.98
    dta['beta^t'] = dta['beta']**dta['t']
    dta['beta^t_reward'] = dta['beta^t'] * dta['reward']

    return dta

# %%


def run_sim_episode(p, o=10):
    allEpisodes = pd.DataFrame()
    for e in range(1000):
        dta = run_sim(p, o)
        dta.insert(0, 'episode', [e] * len(dta), True)
        allEpisodes = allEpisodes.append(dta, ignore_index=True)
    return allEpisodes
# %%


def run_sim_states(p):
    allStates = pd.DataFrame()
    for o in range(1, 20):
        dta = run_sim_episode(p, o)
        dta.insert(0, 'initial', [o - 10] * len(dta), True)
        allStates = allStates.append(dta, ignore_index=True)
    return allStates
# %%


class pipeline:
    def __init__(self, p, dir):
        try:
            self.dta = joblib.load(f'dta_{dir}.pkl')
            self.allEpisodes = joblib.load(f'allEpisodes_{dir}.pkl')
            self.allStates = joblib.load(f'allStates_{dir}.pkl')
        except:
            self.dta = run_sim(p)
            joblib.dump(self.dta, f'dta_{dir}.pkl')
            self.allEpisodes = run_sim_episode(p)
            joblib.dump(self.allEpisodes, f'allEpisodes_{dir}.pkl')
            self.allStates = run_sim_states(p)
            joblib.dump(self.allStates, f'allStates_{dir}.pkl')
        self.perc = [.01, 0.05, .1, .25, .5, .75, .9, .95, .99]

    def print_pipeline(self):
        print(f'Table 2.1:\n')
        display(self.dta)
        print(
            f'\nTotal discounted reward = {np.sum(self.dta["beta^t_reward"])}')
        print('\n' + '-' * 50 + '\n')
        print('Figure 2.2:\n')
        self.allEpisodes[['episode', 'beta^t_reward']
                         ].groupby(by='episode').sum().hist(grid=False)
        plt.pause(1)
        print('\n' + '-' * 50 + '\n')
        print('Table 2.4:\n')
        display(self.allEpisodes[['episode', 'beta^t_reward']].groupby(
            by='episode').sum().describe(self.perc))
        print('\n' + '-' * 50 + '\n')
        print('Table 2.6:\n')
        temp = self.allStates[['initial', 'episode', 'beta^t_reward']].groupby(
            by=['initial', 'episode']).sum().droplevel(level='episode')
        display(temp.groupby('initial').describe())
        print('\n' + '-' * 50 + '\n')
        print('Figure 2.3:\n')
        plt.pause(1)
        plt.scatter(x=temp.groupby('initial').mean().index,
                    y=temp.groupby('initial').mean().values)
        plt.xticks(list(range(-9, 10)))
        plt.xlabel('Initial State, s0')
        plt.ylabel('Value Function, v(s0)')


# %% [markdown]
# 1\) Under the “always try to go right” policy:
#
# rerun the code and recreate table 2.1, 2.4, and 2.6
# recreate figures 2.2 and 2.3
# construct the transition probability matrix
# %%
always_right = pipeline(1, 'right')
always_right.print_pipeline()
# %% [markdown]
# 2\) Under the “always try to go left policy:
#
# rerun the code and recreate table 2.1, 2.4, and 2.6
# recreate figures 2.2 and 2.3
# construct the transition probability matrix
# %%
always_left = pipeline(0, 'left')
always_left.print_pipeline()
# %% [markdown]
# 3) Under a policy that “tries to go left” with $50%$ probability and “tries to go right” with $50%$ probability:
#
# rerun the code and recreate table 2.1, 2.4, and 2.6
# recreate figures 2.2 and 2.3
# construct the transition probability matrix
# %%
half_half = pipeline(0.5, 'half')
half_half.print_pipeline()
