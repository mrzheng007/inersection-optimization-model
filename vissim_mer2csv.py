import csv
import pandas as pd


def read_mer_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def convert_mer_to_csv(mer_file_path):
    # Read the .mer file
    data = read_mer_file(mer_file_path)

    # Assuming the .mer file is structured in a way where each line is a row
    # and values are separated by a delimiter (e.g., comma, tab). Adjust as needed.
    parsed_data = [line.strip().split(';')[0:-1] for line in data if ';' in line]

    # Change the parsed data to a Dataframe
    head_list = ['DataC.P.', 't(enter)', 't(leave)', 'VehNo', 'Type', 'Line', 'v[km/h]', 'a[m/s^2]', 'Occ', 'Pers', 'tQueue', 'VehLength[m]']
    df = pd.DataFrame(parsed_data, columns=head_list)
    # change type
    df[['VehNo', 'Type', 'Line', 'Pers']] = df[['VehNo', 'Type', 'Line', 'Pers']].astype(int)
    df[['t(enter)', 't(leave)', 'v[km/h]', 'a[m/s^2]', 'Occ', 'tQueue', 'VehLength[m]']] = df[['t(enter)', 't(leave)', 'v[km/h]', 'a[m/s^2]', 'Occ', 'tQueue', 'VehLength[m]']].astype(float)
    return df

def Last_volume(x):
    volume_last = x['all_volume']-x['volume_L']-x['volume_T']-x['volume_TT']
    return volume_last

def Volume_approach_ltr(data, dcs, c):
    data_collection = dcs
    cycle = c
    n = 3600 / cycle
    data_s = data[(data['DataC.P.'].isin(data_collection))&(data['t(leave)'] == -1)]
    data_s['cycle'] = data_s['t(enter)'].apply(lambda x: int(x/cycle) + 1)
    data_s['all_volume'] = 1
    group_list = ['cycle']
    df_group = data_s.groupby(group_list, as_index=False).agg({'all_volume': 'sum'})

    df_group['all_volume'] = df_group['all_volume'].apply(lambda x: int(x * n))
    df_group['volume_L'] = df_group['all_volume'].apply(lambda x: int(x*0.15))
    df_group['volume_T'] = df_group['all_volume'].apply(lambda x: int(x * 0.45))
    df_group['volume_TT'] = df_group['all_volume'].apply(lambda x: int(x * 0.25))
    df_group['volume_R'] = df_group.apply(lambda x: Last_volume(x), axis=1)

    df_group['Saturation_L'] = df_group['volume_L'].apply(lambda x: round(x / 1700, 2))
    df_group['Saturation_T'] = df_group['volume_T'].apply(lambda x: round(x / 1800, 2))
    df_group['Saturation_TR'] = df_group.apply(lambda x: round((x['volume_TT']+x['volume_R'])/1800, 2), axis=1)

    return df_group


# Example usage
mer_file_path = r'F:\program\Thomas\2_vehqueue\7_vissimmodel\OptimizationModel_011.mer'
csv_file_path = r'F:\program\Thomas\2_vehqueue\7_vissimmodel\OptimizationModel_011.csv'
df = convert_mer_to_csv(mer_file_path)
df.to_csv(csv_file_path, index=False)
# print(df)
# print(df.dtypes)
eb_8_list = ['25', '30', '70', '63']
df_eb_8 = Volume_approach_ltr(df, eb_8_list, 158)
df_eb_8.to_csv(r'F:\program\Thomas\2_vehqueue\7_vissimmodel\EB_8.csv', index=False)
print(df_eb_8)
