import numpy as np
import pandas as pd


class garden_env:
    def __init__(self, W, B, A, S, P_weather, P_bunny, no_term=True):
        self.W = W
        self.B = B
        self.A = A
        self.S = S
        self.P_weather = P_weather
        self.P_bunny = P_bunny
        self.no_term = no_term
        self.state = None
        self.reward = None
        self.P_bw = self.create_P_bw()
        self.P_s = self.create_P_s()
        self.P_ind, self.P = self.create_P_final()
        self.R = self.create_R_final()
        self.reset_env()

    # Method to create index (weather, bunny)

    def create_P_bw(self):
        ind = [f"({x}, {y})" for y in self.P_bunny.index for x in self.P_weather.index]
        P_bw = pd.DataFrame(columns=ind, index=ind)

        # Populating the bunny/weather matrix
        for row in P_bw.index:
            for col in P_bw.columns:
                i = eval(row)
                j = eval(col)
                P_bw.loc[row, col] = (
                    self.P_weather.loc[i[0], j[0]] * self.P_bunny.loc[j[1], j[0]]
                )

        return P_bw

    def create_P_s(self):
        P_s = np.zeros((len(self.A), len(self.S), len(self.S)), dtype=float)

        # Calculating possible states based on the expression for s'
        for i in self.A:
            for j in self.S:
                for k in self.W:
                    s_new = j - 1 + k + i
                    try:
                        P_s[
                            list(self.A).index(i), self.S.index(j), self.S.index(s_new)
                        ] = 1
                    except:
                        continue
        return P_s

    def create_P_final(self):
        ind = [f"({x}, {y})" for y in self.P_bunny.index for x in self.P_weather.index]
        P_ind = [f"({x}, {eval(y)[0]}, {eval(y)[1]})" for y in ind for x in self.S]

        P_df = pd.DataFrame(
            np.zeros(
                (
                    len(self.S) * len(self.W) * len(self.B),
                    len(self.S) * len(self.W) * len(self.B),
                ),
                dtype=float,
            ),
            index=P_ind,
            columns=P_ind,
        )
        P_list = [P_df.copy() for _ in range(len(self.A))]

        for i in range(len(P_list)):
            for j in P_df.index:
                for k in P_df.columns:
                    row = eval(j)
                    col = eval(k)
                    if (
                        np.clip(
                            row[0] - 1 + col[1] + list(self.A)[i],
                            min(self.S),
                            max(self.S),
                        )
                        == col[0]
                    ):
                        P_list[i].loc[j, k] = (
                            self.P_s[i, self.S.index(row[0]), self.S.index(col[0])]
                            * self.P_bw.loc[
                                f"({row[1]}, {row[2]})", f"({col[1]}, {col[2]})"
                            ]
                        )

            # Terminal state conditions - plant dies at saturation -1 and 5
            # term_dry = [eval(x)[0] == min(self.S) for x in P_ind]
            # term_wet = [eval(x)[0] == max(self.S) for x in P_ind]
            # P_list[i].iloc[term_dry, :] = [1] + [0] * (len(term_dry) - 1)
            # P_list[i].iloc[term_wet, :] = [0] * (len(term_wet) - 1) + [1]
            # P_list[i].iloc[term_dry, :] = np.identity(len(P_ind))[term_dry]
            # P_list[i].iloc[term_wet, :] = np.identity(len(P_ind))[term_wet]
            terminal = [eval(x)[0] in [min(self.S), max(self.S)] for x in P_ind]
            P_list[i].iloc[terminal, :] = np.identity(len(P_ind))[terminal]

        P = np.stack((P_list[0], P_list[1], P_list[2]))

        return P_ind, P

    def create_R_final(self):
        act_penalty = {-1: 0.9, 0: 0, 1: 0.2}
        R = np.zeros((len(self.A), len(self.P_ind)))
        for ind_A, i in enumerate(self.A):
            for ind_S, k in enumerate(self.P_ind):
                R[ind_A, ind_S] = (
                    (-((eval(k)[0] - 1) ** 2) + 2 - (10 * eval(k)[2])) * 0.1  - act_penalty[i]
                    if eval(k)[0] not in [min(self.S), max(self.S)]
                    else -100
                )

        return R

    def reset_env(self):
        if self.no_term:
            self.state = np.random.choice(
                [x for x in self.P_ind if eval(x)[0] not in [min(self.S), max(self.S)]]
            )
        else:
            self.state = np.random.choice(self.P_ind)
        # return state

    def step(self, action):
        rand_num = np.random.random()
        P_cur = pd.DataFrame(self.P[action], index=self.P_ind, columns=self.P_ind)
        self.state = pd.Series(self.P_ind)[
            P_cur.loc[self.state, :].cumsum().ge(rand_num).tolist()
        ].iloc[0]
        self.reward = self.R[action, self.P_ind.index(self.state)]
        # return self.state


if __name__ == "__main__":
    W = {0: "Sunny", 1: "Rainy", 2: "Stormy"}
    B = {0: "No_Bunny", 1: "Bunny"}
    A = {-1: "Sand", 0: "Nothing", 1: "Water"}

    # List of possible saturation states
    S = list(range(-1, 6))

    # Weather to weather transition matrix
    P_weather = pd.DataFrame(
        {0: [0.75, 0.3, 0.4], 1: [0.2, 0.5, 0.4], 2: [0.05, 0.2, 0.2]}
    )

    # Weather to bunny transition matrix
    P_bunny = pd.DataFrame({0: [0.2, 0.8], 1: [0.6, 0.4], 2: [0.9, 0.1]})

    env = garden_env(W, B, A, S, P_weather, P_bunny)
    env.step(0)
    env.state
    env.reward
