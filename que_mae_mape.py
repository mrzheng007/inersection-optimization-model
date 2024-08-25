
import os, pandas as pd, itertools
from err_mae_mape import MAE_MAPE_sim


def Queue_Simulator(data, Cycle_time):
    df = data
    df["no"] = [i + 1 for i in range(len(df))]
    df["sign"] = df["Headway"].apply((lambda x: 1 if x > Cycle_time else 0))
    df.loc[(0, 'sign')] = 1
    df_cyc = df[df["sign"] == 1]
    cycle_all = len(df_cyc)
    df_cyc["cycle"] = [i + 1 for i in range(cycle_all)]
    cycle_dic = df_cyc.set_index("cycle")["no"].to_dict()
    df["cycle"] = 1
    for cyc in cycle_dic:
        veh_start = cycle_dic[cyc]
        if cyc == cycle_all:
            veh_end = len(df) + 1
        else:
            veh_end = cycle_dic[cyc + 1]
        df["cycle"] = df.apply(lambda x: cyc if ((x["no"] >= veh_start) and (x["no"] < veh_end)) else x["cycle"], axis=1)

    df_que = pd.DataFrame()
    groups = df.groupby(["cycle"], as_index=False)
    for c_num, group in groups:
        n = len(group)
        group["veh_order"] = [i + 1 for i in range(n)]
        adjust_queue_list = group["adjust_queue"].to_list()
        continue_list = [len(list(group)) for key, group in itertools.groupby(adjust_queue_list) if key == 1]
        # queque_pre_max = max(continue_list)
        queque_pre_all = group["adjust_queue"].sum()
        # queque_pre_last = group[group["adjust_queue"] == 1]["veh_order"].max()
        # group["queue_pre_max"] = queque_pre_max
        group["queue_pre_all"] = queque_pre_all
        # group["queue_pre_last"] = queque_pre_last
        group["queue_num"] = group["queue"].sum()
        df_que = pd.concat([df_que, group])

    # save_list = ['cycle', 'queue_pre_max', 'queue_pre_all', 'queue_pre_last', 'queue_num']
    save_list = ['cycle', 'queue_pre_all', 'queue_num']
    df_c = df_que[save_list].drop_duplicates(keep="first")

    # for i in ('max', 'all', 'last'):
    for i in ['all']:
        queue_str = "queue_pre_" + i
        err_str = "err_" + i
        err_r_str = "err_r_" + i
        MAE_str_a = "MAE_" + i + "_a"
        MAPE_str_a = "MAPE_" + i + "_a"
        MAE_str_b = "MAE_" + i + "_b"
        MAPE_str_b = "MAPE_" + i + "_b"
        df_c[err_str] = df_c.apply((lambda x: abs(x[queue_str] - x["queue_num"])), axis=1)
        df_c[err_r_str] = df_c.apply((lambda x: abs(x[err_str] * 100 / x["queue_num"])), axis=1)
        df_c[MAE_str_a] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", err_str, err_r_str)[2]))
        df_c[MAPE_str_a] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "a", err_str, err_r_str)[3]))
        df_c[MAE_str_b] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", err_str, err_r_str)[2]))
        df_c[MAPE_str_b] = df_c["cycle"].apply((lambda x: MAE_MAPE_sim(df_c, x, "b", err_str, err_r_str)[3]))

    return df_que, df_c