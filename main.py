import pandas as pd

from adjust2_step3_sequential_label_modification import Adjust_que
from adjust_abmn_new import Queue_Abmn_0
from processing import get_df
from ANN_train_step1_binary_classification import ANN_train
from que_mae_mape import Queue_Simulator
from testconnectivity_step2_Fourcategoryclassification import Fourcategoryclassification


# Load the Excel file
file_path = './simulation output.xlsx'
xls = pd.ExcelFile(file_path)

# Load the first sheet into a DataFrame
df = pd.read_excel(xls, sheet_name=0, header=None)
# print(df)

# Function to find the header row
def find_header_row(df, header_name):
    for i, row in df.iterrows():
        if header_name in row.values:
            return i
    return None

# Search for the header row containing 'DataC.P.'
header_row = find_header_row(df, 'DataC.P.')

if header_row is not None:
    # # Set the detected header row as the DataFrame's column names
    # df.columns = df.iloc[header_row]
    # # Drop the initial non-data rows
    # df = df.drop(range(header_row + 1)).reset_index(drop=True)

    # Read the Excel file again, starting from the header row
    df = pd.read_excel(xls, sheet_name=0, header=header_row)

else:
    raise ValueError("Header row with 'DataC.P.' not found")

# Rename the column "a[m/s瞉" to "a[m/s^2]"
df.rename(columns={"a[m/s瞉": "a[m/s^2]"}, inplace=True)

# Add the v[km/h] column
df['v[km/h]'] = df['v[m/s]'] * 3.6

# Select relevant columns
df_selected = df[['DataC.P.', 't(enter)', 't(leave)', 'VehNo', 'Type', 'v[m/s]', 'v[km/h]', 'a[m/s^2]', 'Occ', 'tQueue', 'VehLength[m]']]
# print(df_selected)

# Create and process separate files for each DataC.P. value
for value in df_selected['DataC.P.'].unique():
    if value == 1:
        df_filtered = df_selected[df_selected['DataC.P.'] == value]

        # Sort the filtered DataFrame by 'VehNo' in ascending order
        df_filtered = df_filtered.sort_values(by='VehNo')

        # Process the filtered and sorted DataFrame
        processed_df = get_df(df_filtered)

        # If not reset the index, then in step3, the 'VehNo' column cannot be indexed as it is shifted by one row (not the index)
        # data = processed_df

        # Reset the index to add the default integer column as the new index column (previously the 'VehNo' column is one row shifted)
        data = processed_df.reset_index(drop=False)
        print(data)

        # processed_df.to_csv(f'./file/{value}.csv')
        # # print(f"CSV file for 'DataC.P.={value}' has been created and processed successfully.")
        #
        # # train and test
        # data = pd.read_csv(f'./file/{value}.csv')
        # print(data)


        # Split data into training and testing sets
        train_num = int(data.shape[0] * 0.75)
        data_train = data.iloc[:train_num, :]
        data_test = data.iloc[train_num:, :]

        print(data_train)
        print(data_test)


        sheetnames = ['N1', 'N2', 'E3', 'E4', 'S5', 'S6', 'W7', 'W8']
        Y_SELECTs = ['Type', 'queue']
        sheetname = sheetnames[0]
        Y_SELECT = Y_SELECTs[1]

        # Step1: Binary classification
        df_test = ANN_train(data_train, data_test)
        # df_test.to_csv('./results/ANN_result_{}_{}.csv'.format(sheetname, Y_SELECT))

        # Step2: Four-category classification
        # print(df_train.dtypes)
        df_class4 = Fourcategoryclassification(data_train, data_test)
        # pd.DataFrame(df_class4).to_csv('./results/result_class4_{}_{}.csv'.format(sheetname, Y_SELECT))

        # Step3: Adjust
        # print(len(df_test), len(df_class4))
        df_adjust = Adjust_que(df_test, df_class4)
        # df_adjust = Adjust_pre(df_test, df_class4)
        # df_adjust.to_csv(r'./results/Sequential_adjust_{}.csv'.format(value), index=0, encoding='utf_8_sig')

        # Step4: QLE
        # (1) From # of 1s
        # df_adjust = pd.read_csv(r'./results/Sequential_adjust_{}.csv'.format(value), encoding='utf_8_sig')
        phase_time = 97
        df_que_1, df_err = Queue_Simulator(df_adjust, phase_time)
        print(df_que_1, df_err)
        df_que_1.to_csv(r'./results/queue_1_{}.csv'.format(value), index=0, encoding='utf_8_sig')
        df_err.to_csv(r'./results/queue_err_{}.csv'.format(value), index=0, encoding='utf_8_sig')
        # (2) From # of 0s
        df_que_0 = Queue_Abmn_0(df_que_1)
        df_que_0.to_csv(r'./results/queue_0_{}.csv'.format(value), index=0, encoding='utf_8_sig')

