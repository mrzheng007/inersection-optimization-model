
import os, pandas as pd, itertools

def count_consecutive_duplicates(sequence):
    key_list = [key for key, group in itertools.groupby(sequence)]
    value_list = [len(list(group)) for key, group in itertools.groupby(sequence)]
    return (
     key_list, value_list)


def C_0(c_column, c):
    c_lst = c_column["c_list"]
    queue_lst = c_column["queue_list"]
    C0_index = [index for index, value in enumerate(queue_lst) if value == 0]
    if len(C0_index) == 0:
        que_len = sum(c_lst)
    else:
        df_g = pd.DataFrame({'key':queue_lst,  'value':c_lst})
        df_c0 = df_g[(df_g["key"] == 0) & (df_g["value"] > c)]
        if len(df_c0) == 0:
            if df_g["key"].to_list()[-1] == 0:
                que_len = sum(c_lst) - c_lst[-1]
            else:
                que_len = sum(c_lst)
        else:
            index = df_c0.index.min()
            que_len = sum(c_lst[0:index])
    return que_len

def MAE_MAPE_sim(data, x, err_str, err_r_str):
    c = x["cycle"]
    # print(c)
    n_c = 43
    if c <= (len(data) - n_c + 1):
        c_start = c
        c_end = c + n_c
        df_peak = data[(data["cycle"] >= c_start) & (data["cycle"] < c_end)]
        err_sum = df_peak[err_str].sum()
        err_r_sum = df_peak[err_r_str].sum()
        MAE = df_peak[err_str].sum() / n_c
        MAPE = df_peak[err_r_str].sum() / n_c
    else:
        err_sum = ""
        err_r_sum = ""
        MAE = ""
        MAPE = ""
    return (err_sum, err_r_sum, MAE, MAPE)

def Queue_Abmn_0(data_process):
    df = data_process
    data_list = []
    groups = df.groupby(["cycle"], as_index=False)
    for cycle, group in groups:
        # print(cycle)
        n = len(group)
        queue_num = group["queue_num"].to_list()[0]
        pre_queue_seq = group["adjust_queue"].to_list()
        queue_list, c_list = count_consecutive_duplicates(pre_queue_seq)
        c_sum = sum(c_list)
        queue_dic = {'cycle': int(cycle[0]),
         'queue_list': queue_list, 
         'c_list': c_list, 
         'veh_num': n, 
         'c_sum': c_sum, 
         'queue_num': queue_num}
        data_list.append(queue_dic)

    data = pd.DataFrame(data_list)
    data["queue_1"] = data.apply((lambda x: C_0(x, 1)), axis=1)
    data["queue_2"] = data.apply((lambda x: C_0(x, 2)), axis=1)
    data["queue_3"] = data.apply((lambda x: C_0(x, 3)), axis=1)
    data["queue_4"] = data.apply((lambda x: C_0(x, 4)), axis=1)
    data["queue_5"] = data.apply((lambda x: C_0(x, 5)), axis=1)
    data["queue_6"] = data.apply((lambda x: C_0(x, 6)), axis=1)

    for i in range(6):
        queue_str = "queue_" + str(i + 1)
        err_str = "err_" + str(i + 1)
        err_r_str = "err_r_" + str(i + 1)
        err_sum_str = "err_sum_" + str(i + 1)
        err_r_sum_str = "err_r_sum_" + str(i + 1)
        MAE_str = "MAE_" + str(i + 1)
        MAPE_str = "MAPE_" + str(i + 1)
        # print(data)
        # data['cycle'] = data['cycle'].apply(lambda x: x).astype(int)
        data[err_str] = data.apply((lambda x: abs(x[queue_str] - x["queue_num"])), axis=1)
        data[err_r_str] = data.apply((lambda x: abs(x[err_str] * 100 / x["queue_num"])), axis=1)
        data[MAE_str] = data.apply((lambda x: MAE_MAPE_sim(data, x, err_str, err_r_str)[2]), axis=1)
        data[MAPE_str] = data.apply((lambda x: MAE_MAPE_sim(data, x, err_str, err_r_str)[3]), axis=1)

    return data
