# %%
import numpy as np
import pandas as pd

# %%
w = {0: 'Sunny', 1: 'Rainy', 2: 'Stormy'}
b = {0: 'No_Bunny', 1: 'Bunny'}
a = {-1: 'Sand', 0: 'Nothing', 1: 'Water'}

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
s = list(range(-1, 6))
P_s = np.zeros((len(a), len(s), len(s)), dtype=float)
for i in a:
    for j in s:
        for k in w:
            s_new = j - 1 + k + i
            try:
                P_s[list(a).index(i), s.index(j), s.index(s_new)] = 1
            except:
                continue
# %%
P_ind = [f'({x}, {eval(y)[0]}, {eval(y)[1]})' for y in ind for x in s]

P_df = pd.DataFrame(
    np.zeros(
        (
            len(s) * len(w) * len(b),
            len(s) * len(w) * len(b)),
        dtype=float),
    index=P_ind, columns=P_ind)
P_list = [P_df.copy() for x in range(len(a))]

# %%

for i in range(len(P_list)):
    for j in P_df.index:
        for k in P_df.columns:
            row = eval(j)
            col = eval(k)
            if np.clip(row[0] - 1 + col[1] + list(a)[i], min(s), max(s)) == col[0]:
                P_list[i].loc[j, k] = P_s[i, s.index(row[0]), s.index(col[0])] * \
                    P_bw.loc[f'({row[1]}, {row[2]})', f'({col[1]}, {col[2]})']

# %%
P = np.stack((P_list[0], P_list[1], P_list[2]))