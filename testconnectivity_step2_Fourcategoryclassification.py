import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.layers import Input, Dense
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from fuzzy_layer import FuzzyLayer
from defuzzy_layer import DefuzzyLayer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import seaborn as sns

from sklearn.model_selection import GridSearchCV
import time
import joblib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(30)

def relabel(df):
    # Reset the index as in naming the new labels, the below for loop is operated based on the index column started
    # with '0', while the original test_data's index column is started from '14838' insteaf of '0'
    df = df.reset_index(drop=False)
    df['processed_label'] = 0  # 初始化新列
    print(df)

    for i in range(1, len(df)):
        current_value = df.at[i, 'queue']
        previous_value = df.at[i - 1, 'queue']

        if current_value == 0 and previous_value == 0:
            df.at[i, 'processed_label'] = 0
        elif current_value == 0 and previous_value == 1:
            df.at[i, 'processed_label'] = 1
        elif current_value == 1 and previous_value == 0:
            df.at[i, 'processed_label'] = 2
        elif current_value == 1 and previous_value == 1:
            df.at[i, 'processed_label'] = 3

    return df

# def Processed_label(x):
#     current_value = x['queue']
#     previous_value = x['previous_queue']
#     processed_label = 0
#     if current_value == 0 and previous_value == 0:
#         processed_label = 0
#     elif current_value == 0 and previous_value == 1:
#         processed_label = 1
#     elif current_value == 1 and previous_value == 0:
#         processed_label = 2
#     elif current_value == 1 and previous_value == 1:
#         processed_label = 3
#     return processed_label
#
# def relabel(df):
#     # df.loc[0, 'processed_label'] = 0  # 初始化新列
#     # print(len(df))
#     df['previous_queue'] = df['queue'].shift(1).fillna(0).astype(int)
#     # print(df.columns)
#     # df['processed_label'] = 0
#     df['processed_label'] = df.apply(lambda x: Processed_label(x), axis=1)
#     df.at[0, 'processed_label'] = 0  # Set the first row of processed_label to 0
#     # print(len(df))
#     return df


def Fourcategoryclassification(train_data, test_data):
    sheetnames = ['N1', 'N2', 'E3', 'E4', 'S5', 'S6', 'W7', 'W8']
    Y_SELECTs = ['Type', 'queue']

    sheetname = sheetnames[0]
    Y_SELECT = Y_SELECTs[1]

    columns_to_diff = ["Headway", "Rearway", "Speed_1", "Speed_2", "Occ_1", "Occ_2", "Accel_1", "Accel_2"]

    train_data_diff = train_data[columns_to_diff].diff().fillna(0)
    train_data_diff = train_data.assign(
        Headway_diff=train_data_diff['Headway'],
        Rearway_diff=train_data_diff['Rearway'],
        Speed_1_diff=train_data_diff['Speed_1'],
        Speed_2_diff=train_data_diff['Speed_2'],
        Occ_1_diff=train_data_diff['Occ_1'],
        Occ_2_diff=train_data_diff['Occ_2'],
        Accel_1_diff=train_data_diff['Accel_1'],
        Accel_2_diff=train_data_diff['Accel_2'])

    # print(train_data_diff.dtypes)
    train_data_diff = relabel(train_data_diff)
    # train_data_diff.to_csv('./file/train_diff_{}_{}.csv'.format(sheetname, Y_SELECT))

    test_data_diff = test_data[columns_to_diff].diff().fillna(0)
    test_data_diff = test_data.assign(
        Headway_diff=test_data_diff['Headway'],
        Rearway_diff=test_data_diff['Rearway'],
        Speed_1_diff=test_data_diff['Speed_1'],
        Speed_2_diff=test_data_diff['Speed_2'],
        Occ_1_diff=test_data_diff['Occ_1'],
        Occ_2_diff=test_data_diff['Occ_2'],
        Accel_1_diff=test_data_diff['Accel_1'],
        Accel_2_diff=test_data_diff['Accel_2'])

    # print(len(test_data_diff))
    test_data_diff = relabel(test_data_diff) # .at方法数据得到2倍？
    # test_data_diff.to_csv('./file/test_diff_{}_{}.csv'.format(sheetname, Y_SELECT))
    # print(len(test_data_diff))


    train_data = train_data_diff
    test_data = test_data_diff
    # print(train_data.columns)

    x_train = train_data[["Headway_diff", "Rearway_diff", "Speed_1_diff", "Speed_2_diff", "Occ_1_diff", "Occ_2_diff", "Accel_1_diff", "Accel_2_diff"]].values
    y_train = train_data["processed_label"].values
    x_test = test_data[["Headway_diff", "Rearway_diff", "Speed_1_diff", "Speed_2_diff", "Occ_1_diff", "Occ_2_diff", "Accel_1_diff", "Accel_2_diff"]].values
    y_test = test_data["processed_label"].values

    print(x_train.shape)

    scaler = MinMaxScaler() #MinMaxScaler  StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)



    def fuzzy_model(h1=128, h2=16, h3=128):#
        input_layer = Input(shape=(8,))
        dense_1 = Dense(h1)(input_layer)
        fuzzy_layer = FuzzyLayer(h2)(dense_1)
        defuzzy_layer = DefuzzyLayer(h3)(fuzzy_layer)
        output_layer = Dense(4, activation='softmax')(defuzzy_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model





    """
    以最优参数训练fuzzy_model需注释掉 def fuzzy_model的model.compile
    """
    model = fuzzy_model()

    # Check GPU availability and print a message
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU")
    else:
        print("Using CPU")

    model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), batch_size=256,
                    verbose=2)  # 100

    # tr_acc = history.history['accuracy']
    # val_acc= history.history['val_accuracy']
    # tr_loss = history.history['loss']
    # val_loss= history.history['val_loss']

    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)

    y_prob = model.predict(X_test)#[:, 1]
    y_test = np.argmax(y_test, axis=1)
    # print(len(y_pred))
    test_data['result'] = y_pred
    # pd.DataFrame(test_data).to_csv('./results/result_class4_{}_{}.csv'.format(sheetname, Y_SELECT))

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    print(test_data)

    return test_data


