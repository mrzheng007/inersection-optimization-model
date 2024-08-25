import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.regularizers import l2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# 定义一个函数，创建Keras模型，以便使用GridSearchCV搜索最优参数
def create_ANN_model_for_grid_search(input_shape, hidden_layer_sizes=(200,200), activation='sigmoid', l2_regularization=0.0):
    model = Sequential()

    for i, units in enumerate(hidden_layer_sizes):
        if i == 0:
            model.add(
                Dense(units, activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_regularization)))
        else:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_regularization)))

    model.add(Dense(1, activation='sigmoid'))  # 二分类问题的输出层

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def ANN_train(df_train, df_test):
    # 数据处理
    sheetnames = ['N1', 'N2', 'E3', 'E4', 'S5', 'S6', 'W7', 'W8']
    Y_SELECTs = ['Type', 'queue']
    #
    sheetname = sheetnames[0]
    Y_SELECT = Y_SELECTs[1]
    #
    # test_path = r'../SVM_occ & acceleration/test_data/test_{}_{}.csv'.format(sheetname, Y_SELECT)
    # train_path = r'../SVM_occ & acceleration/train_data/train_{}_{}.csv'.format(sheetname, Y_SELECT)
    #
    # df_train = pd.read_csv(train_path, index_col='VehNo', dtype=np.float32)

    X_train = df_train[['Headway', 'Rearway', 'Speed_1', 'Speed_2', 'Occ_1', 'Occ_2', 'Accel_1', 'Accel_2']].astype(np.float32)
    y_train = df_train[Y_SELECT].astype(int)

    # df_test = pd.read_csv(test_path, index_col='VehNo', dtype=np.float32)
    X_test = df_test[['Headway', 'Rearway', 'Speed_1', 'Speed_2', 'Occ_1', 'Occ_2', 'Accel_1', 'Accel_2']].astype(np.float32)
    y_test = df_test[Y_SELECT].astype(int)

    # 获取数据的输入形状
    input_shape = X_train.shape[1:]

    # Create and train the model
    model = create_ANN_model_for_grid_search(input_shape)

    # Check GPU availability and print a message
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU")
    else:
        print("Using CPU")

    # history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=grid_search.best_params_['epochs'], batch_size=32)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

    # 使用最优模型对测试集进行预测
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 将预测值添加到df_test中
    df_test['pre_{}'.format(Y_SELECT)] = y_pred_binary


    # 计算最优模型的准确率
    model_accuracy = accuracy_score(y_test, y_pred_binary)
    print("最优模型在测试集上的 准确率:", model_accuracy)

    print(df_test)
    return df_test

