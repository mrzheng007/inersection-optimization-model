# uncompyle6 version 3.9.1
# Python bytecode version base 3.6 (3379)
# Decompiled from: Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 10:22:32) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: F:\program\Thomas\2_vehqueue\2_code\queue_EDM.py
# Compiled at: 2024-04-15 00:47:07
# Size of source mod 2**32: 4950 bytes
import statistics, pandas as pd
from err_mae_mape import MAE_MAPE_sim

def Queue_EDM_Simulator(df_process):
    data = df_process
    data["Headway_1"] = data["Headway"].apply((lambda x: x))
    df = pd.DataFrame()
    groups = data.groupby(["cycle"], as_index=False)
    for C, group in groups:
        N = len(group)
        group["veh_order"] = [i + 1 for i in range(N)]
        if N > 3:
            index = group.index[0]
            group.loc[(index, "Headway_1")] = 2
            D_list = []
            for t in range(1, N - 1):
                A_list = []
                B_list = []
                AB_list = []
                for i in range(t):
                    for j in range(i + 1, t + 1):
                        hi = group.loc[(index + i, "Headway_1")]
                        hj = group.loc[(index + j, "Headway_1")]
                        a = abs(hi - hj)
                        A_list.append(a)

                for m in range(t + 1, N - 1):
                    for n in range(m + 1, N):
                        hm = group.loc[(index + m, "Headway_1")]
                        hn = group.loc[(index + n, "Headway_1")]
                        b = abs(hm - hn)
                        B_list.append(b)

                u_list = range(1, t + 1)
                v_list = range(t + 1, N)
                for u in u_list:
                    for v in v_list:
                        hu = group.loc[(index + u, "Headway_1")]
                        hv = group.loc[(index + v, "Headway_1")]
                        ab = abs(hu - hv)
                        AB_list.append(ab)

                Da = statistics.median(A_list)
                if len(B_list):
                    Db = statistics.median(B_list)
                else:
                    Db = 0
                Dab = statistics.median(AB_list)
                p = t - 1
                q = N - 1 - t
                d = p * q * (2 * Dab - Da - Db) / (p + q)
                D_list.append(d)

            df_D = pd.Series(D_list)
            max_t = df_D.idxmax()
            no_max = max_t + 1
            Q = int(group[group["veh_order"] < no_max]["Headway_1"].sum() / 2)
            group["max_t"] = max_t
            group["max_D"] = max(D_list)
            group["Q"] = Q
            group["queue_num"] = group["queue"].sum()
            df = df.append(group)

    df_c = df[['cycle', 'queue_num', 'max_t', 'max_D', 'Q']].drop_duplicates(keep="first")
    df_c.sort_values("cycle", inplace=True)
    df_c["err"] = df_c.apply((lambda x: abs(x["Q"] - x["queue_num"])), axis=1)
    df_c["err_r"] = df_c.apply((lambda x: abs(x["err"] * 100 / x["queue_num"])), axis=1)
    # df_c["err_sum_a"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", "err", "err_r")[0]))
    # df_c["err_r_sum_a"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", "err", "err_r")[1]))
    # df_c["MAE_a"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", "err", "err_r")[2]))
    # df_c["MAPE_a"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", "err", "err_r")[3]))
    # df_c["err_sum_b"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", "err", "err_r")[0]))
    # df_c["err_r_sum_b"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", "err", "err_r")[1]))
    # df_c["MAE_b"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", "err", "err_r")[2]))
    # df_c["MAPE_b"] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", "err", "err_r")[3]))
    return df_c


if __name__ == "__main__":
    data = pd.read_csv(r"F:\program\Thomas\2_vehqueue\1_data\240414\test_E2_queue_timestamp.csv")
    df = Queue_EDM_Simulator(data)
    df.to_csv(r"F:\program\Thomas\2_vehqueue\1_data\240414\edm_result0414\test_E2_edm_result.csv")
