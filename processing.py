import pandas as pd

def get_df(df):
    df_N1 = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    df_tmp = df_N1.copy()
    df_tmp = df_tmp[df_N1.iloc[:, 0] != -1].set_index('VehNo')
    v1 = df_tmp['v[km/h]'].copy()
    Occ1 = df_tmp['Occ'].copy()
    a1 = df_tmp['a[m/s^2]'].copy()
    tmp = df_tmp.iloc[:, 0]
    df_tmp['Headway'] = tmp - tmp.shift(1)

    df_tmp1 = df_N1.copy()
    df_tmp1 = df_tmp1[df_N1.iloc[:, 1] != -1].set_index('VehNo')
    v2 = df_tmp1['v[km/h]'].copy()
    Occ2 = df_tmp1['Occ'].copy()
    a2 = df_tmp1['a[m/s^2]'].copy()
    tmp1 = df_tmp1.iloc[:, 1]

    df_tmp['Rearway'] = tmp1 - tmp1.shift(1)
    df_tmp[df_tmp != df_tmp] = 0

    queue_bool = df_tmp['tQueue'].copy()
    queue_bool[df_tmp['tQueue'] != 0] = 1
    df_tmp['queue'] = queue_bool

    type_bool = df_tmp['Type'].copy()
    type_bool[df_tmp['Type'] == 100] = 0
    type_bool[df_tmp['Type'] == 200] = 1
    df_tmp['Type'] = type_bool
    df_tmp['Speed_1'] = v1
    df_tmp['Speed_2'] = v2
    df_tmp['Occ_1'] = Occ1
    df_tmp['Occ_2'] = Occ2
    df_tmp['Accel_1'] = a1
    df_tmp['Accel_2'] = a2

    return df_tmp[['Headway', 'Rearway', 'Speed_1', 'Speed_2', 'Occ_1', 'Occ_2', 'Accel_1', 'Accel_2', 'Type', 'queue', ]]


