# %%
import numpy as np
import pandas as pd

# %%
W = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
B = {0: 'No_Bunny', 1: 'Bunny'}
A = {-1: 'Sand', 0: 'Nothing', 1: 'Water'}

P_weather = pd.DataFrame(
    {
        0: [0.75, 0.3, 0.4],
        1: [0.2, 0.5, 0.4],
        2: [0.05, 0.2, 0.2]
    })
P_bunny = pd.DataFrame(
    {
        0: [0.2, 0.8],
        1: [0.6, 0.4],
        2: [0.9, 0.1]
    })
# %%
ind = [f'({x}, {y})' for y in P_bunny.index for x in P_weather.index]
P_bw = pd.DataFrame(columns=ind, index=ind)
# pattern = re.compile(r'[\(\) ]')
for row in P_bw.index:
    for col in P_bw.columns:
        i = eval(row)
        j = eval(col)
        P_bw.loc[row, col] = P_weather.loc[i[0], j[0]] * \
            P_bunny.loc[j[1], j[0]]
# %% [markdown]
# $s' = s - 1 + w' + a$
# %%
S = list(range(-1, 6))
P_s = np.zeros((len(A), len(S), len(S)), dtype=float)
for i in A:
    for j in S:
        for k in W:
            s_new = j - 1 + k + i
            try:
                P_s[list(A).index(i), S.index(j), S.index(s_new)] = 1
            except:
                continue
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
    # add terminal state conditions here
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
            R[i, j, k] = (-(curr[0] - 1)**2 + 2 - (10 * curr[2]) - abs(list(A)[i])) * 0.1
    term_dry = [eval(x)[0] == min(S) for x in P_ind]
    term_wet = [eval(x)[0] == max(S) for x in P_ind]
    R[i, term_dry, :] = [-np.inf] * len(term_dry)
    R[i, term_wet, :] = [-np.inf] * len(term_dry)
