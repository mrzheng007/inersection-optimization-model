
import numpy as np

def MAE_MAPE_sim(df, c, s, err_str, err_r_str):
    n_c = 43
    if s == "a":
        if c <= len(df) - n_c + 1:
            c_start = c
            c_end = c + n_c
            df_peak = df[(df["cycle"] >= c_start) & (df["cycle"] < c_end)]
            # print(df_peak)
            err_sum = df_peak[err_str].sum()
            err_r_sum = df_peak[err_r_str].sum()
            MAE = round(err_sum/n_c, 2)
            MAPE = round(err_r_sum/n_c, 2)
            result_list = [err_sum, err_r_sum, MAE, MAPE]
        else:
            result_list = [np.NAN, np.NAN, np.NAN, np.NAN]
    else:
        if c >= n_c:
            c_start = c - n_c
            c_end = c
            df_peak = df[(df["cycle"] > c_start) & (df["cycle"] <= c_end)]
            err_sum = df_peak[err_str].sum()
            err_r_sum = df_peak[err_r_str].sum()
            MAE = round(err_sum / n_c, 2)
            MAPE = round(err_r_sum / n_c, 2)
            result_list = [err_sum, err_r_sum, MAE, MAPE]
        else:
            result_list = [np.NAN, np.NAN, np.NAN, np.NAN]
    return result_list
