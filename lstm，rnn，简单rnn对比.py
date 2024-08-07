# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:26:02 2024

@author: 罗鑫
"""

from sklearn import datasets, svm, metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestClassifier  # 导入RandomForestClassifier  
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor  
from sklearn.svm import SVR  
from sklearn.neural_network import MLPRegressor  
import pandas as pd  
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys
print(sys.version)
# 加载数据
data = pd.read_excel(r"D:\统计建模1\训练模型文件.xlsx",
                  ) 
# 获取列名
columns = data.columns
data.set_index(data.columns[0], inplace=True)
# 选择第 7 到第 11 列
train = data[0:252]
test = data[252:278]
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

df_train = train
df_test = test
df_train
columns=df_train.columns
#训练集


#测试集
data1=df_train.iloc[:,:10]
import matplotlib.pyplot as plt

# 假定df_train是您的训练集DataFrame

# 绘制每一列的时序图
plt.figure(figsize=(15, 14))  # 设置整个画布的大小

# 得到DataFrame的所有列名
columns = data1.columns

# 对每一个特征（列）进行循环绘图
for i, column in enumerate(columns):
    plt.subplot(3, 4, i+1)  # 创建一个新的子图，4行2列的格局中的第i+1个位置
    plt.plot(data[column], color='g', alpha=0.7)  # 增加透明度  # 绘制时序图
    plt.title(f'{column}时序图')  # 设置标题，利用了f-string来插入变量的值
    plt.grid(True)  # 开启网格

# 调整子图的间距
plt.tight_layout()
plt.show()


#录波器异常值处理
def hampel(vals_orig, k=7, t0=3):
    vals_filt = np.copy(vals_orig)
    outliers_indices = []
    n = len(vals_orig)

    for i in range(k, n - k):
        window = vals_orig[i - k:i + k + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        if np.abs(vals_orig[i] - median) > t0 * mad:
            vals_filt[i] = median
            outliers_indices.append(i)

    return vals_filt, outliers_indices
filtered_data, outliers_indices = hampel(df_train['商品房均价'])

go_over = df_train['商品房均价']
df_train['商品房均价'] = filtered_data

plt.figure(figsize=(20, 7))
plt.subplot(2, 1, 1)
plt.plot(go_over, color='g',  alpha=0.3)
plt.title('商品房均价异常时序图')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(df_train['商品房均价'], color='g',  alpha=0.3)
plt.title('商品房均价Hampel时序图')
plt.grid(True)
plt.show()
c=df_train.columns

from sklearn.preprocessing import MinMaxScaler
#标准化数据
def normalize_dataframe(train_df, test_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df)  # 在训练集上拟合归一化模型
    train_data = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns, index = df_train.index)
    test_data = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index = df_test.index)
    return train_data, test_data

data_train, data_test = normalize_dataframe(df_train, df_test)
data_train


#滑动窗口
def prepare_data(data, win_size, target_feature_idx, exclude_features=[]):
    num_features = data.shape[1] - len(exclude_features)  # 更新特征数量
    X = []  # 用于存储输入特征的列表
    y = []  # 用于存储目标值的列表
    # 遍历数据，形成样本窗口
    for i in range(len(data) - win_size):
        temp_x = []
        for j in range(data.shape[1]):
            if j != target_feature_idx and j not in exclude_features:
                temp_x.append(data[i:i + win_size, j])  # 提取窗口大小范围内的输入特征
        temp_y = data[i + win_size, target_feature_idx]  # 提取对应的目标值
        X.append(temp_x)
        y.append(temp_y)
    # 转换列表为 NumPy 数组，并调整维度顺序
    X = np.asarray(X).transpose(0, 2, 1)  
    y = np.asarray(y)
    return X, y
win_size = 12
target_feature_idx = 8 # 指定待预测特征
exclude_features = [8] #  需要删除的自变量特征也就是不把待预测的特征纳入输入特征进行时间窗口划分
train_x, train_y = prepare_data(data_train.values, win_size, target_feature_idx)
test_x, test_y = prepare_data(data_test.values, win_size, target_feature_idx)
print("训练集形状:", train_x.shape, train_y.shape)
print("测试集形状:", test_x.shape, test_y.shape)

def prepare_data(data, win_size, target_feature_idx):
    num_features = data.shape[1]
    X = []  
    y = []  
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size, :]  
        temp_y = data[i + win_size, target_feature_idx]  
        X.append(temp_x)
        y.append(temp_y)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

win_size = 12 # 时间窗口
target_feature_idx = 8 # 指定待预测特征列
train_x, train_y = prepare_data(data_train.values, win_size, target_feature_idx)
test_x, test_y = prepare_data(data_test.values, win_size, target_feature_idx)
print("训练集形状:", train_x.shape, train_y.shape)
print("测试集形状:", test_x.shape, test_y.shape)


#最好模型一lstm
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense, Dropout  
import tensorflow as tf
# 模型构建
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))#正则化，防止过拟合
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # 添加一个 dropout 层 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 编译模型
model.compile(loss='mse', optimizer = 'adam')
#pso降维数据，kpso。
# 模型拟合,
history = model.fit(train_x, train_y, epochs=25, batch_size=32,validation_data=(test_x, test_y))
model.summary()#模型总结
plt.figure()
plt.plot(history.history['loss'], c='b', label = 'loss')
plt.plot(history.history['val_loss'], c='g', label = 'val_loss')
plt.legend()
plt.show()










from sklearn import metrics
y_pred = model.predict(test_x)
# 计算均方误差（MSE）
mse = metrics.mean_squared_error(test_y, np.array([i for arr in y_pred for i in arr]))
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
# 计算平均绝对误差（MAE）
mae = metrics.mean_absolute_error(test_y, np.array([i for arr in y_pred for i in arr]))
from sklearn.metrics import r2_score # 拟合优度
r2 = r2_score(test_y, np.array([i for arr in y_pred for i in arr]))
print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
#print("拟合优度:", r2)
#滑动窗口包含预测特征
def predict_future(model, initial_sequence, steps):
    predicted_values = []  # 存储预测结果
    current_sequence = initial_sequence.copy()  # 初始序列
    for i in range(steps):
        # 使用模型进行单步预测
        predicted_value = model.predict(current_sequence.reshape(1, initial_sequence.shape[0], initial_sequence.shape[1]))
        # 将预测结果添加到列表中
        predicted_values.append(predicted_value[0][0])
        # 更新当前序列，删除第一个时间步并将预测值添加到最后一个时间步
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = predicted_value
    return predicted_values

# 使用该函数进行预测
steps_to_predict =  12 # 要预测的步数
predicted_values = predict_future(model, test_x[-1], steps_to_predict)

train_max = np.max(df_train['商品房均价'])
train_min = np.min(df_train['商品房均价'])

series1 = np.array(predicted_values)*(train_max-train_min)+train_min
print(series1)







#rnn模型二
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import GRU, Dense, Dropout  
# 模型构建  
model = Sequential()  
# 注意：这里使用GRU替代了LSTM  
model.add(GRU(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))  
model.add(Dropout(0.2))  # 正则化，防止过拟合  
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.2))  # 添加一个 dropout 层  
model.add(Dense(32, activation='relu'))  
model.add(Dense(1))  
  
# 编译模型  
model.compile(loss='mse', optimizer='adam')  
  
# 模型拟合  
history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_data=(test_x, test_y))

#滑动窗口包含预测特征
def predict_future(model, initial_sequence, steps):
    predicted_values = []  # 存储预测结果
    current_sequence = initial_sequence.copy()  # 初始序列
    for i in range(steps):
        # 使用模型进行单步预测
        predicted_value = model.predict(current_sequence.reshape(1, initial_sequence.shape[0], initial_sequence.shape[1]))
        # 将预测结果添加到列表中
        predicted_values.append(predicted_value[0][0])
        # 更新当前序列，删除第一个时间步并将预测值添加到最后一个时间步
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = predicted_value
    return predicted_values
# 使用该函数进行预测
steps_to_predict =  12 # 要预测的步数
predicted_values = predict_future(model, test_x[-1], steps_to_predict)

train_max = np.max(df_train['商品房均价'])
train_min = np.min(df_train['商品房均价'])

series2 = np.array(predicted_values)*(train_max-train_min)+train_min
print(series2)



#简单rnn模型三
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout  
  
# 模型构建  
model = Sequential()  
# 注意：这里使用SimpleRNN替代了LSTM  
model.add(SimpleRNN(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))  
model.add(Dropout(0.2))  # 正则化，防止过拟合  
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.2))  # 添加一个 dropout 层  
model.add(Dense(32, activation='relu'))  
model.add(Dense(1))  
  
# 编译模型  
model.compile(loss='mse', optimizer='adam')  
  
# 模型拟合  
history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_data=(test_x, test_y))

#滑动窗口包含预测特征
def predict_future(model, initial_sequence, steps):
    predicted_values = []  # 存储预测结果
    current_sequence = initial_sequence.copy()  # 初始序列
    for i in range(steps):
        # 使用模型进行单步预测
        predicted_value = model.predict(current_sequence.reshape(1, initial_sequence.shape[0], initial_sequence.shape[1]))
        # 将预测结果添加到列表中
        predicted_values.append(predicted_value[0][0])
        # 更新当前序列，删除第一个时间步并将预测值添加到最后一个时间步
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = predicted_value
    return predicted_values
# 使用该函数进行预测
steps_to_predict =  12 # 要预测的步数
predicted_values = predict_future(model, test_x[-1], steps_to_predict)

train_max = np.max(df_train['商品房均价'])
train_min = np.min(df_train['商品房均价'])

series3 = np.array(predicted_values)*(train_max-train_min)+train_min
print(series3)




#最终模型1
#加一个k交叉验证--lstm
from tensorflow.keras.models import Sequential    
from tensorflow.keras.layers import LSTM, Dense, Dropout    
import tensorflow as tf  
from sklearn.model_selection import KFold  
import numpy as np  
  
# 假设 train_x, train_y 已经准备好了  
  
# 定义KFold对象，例如我们使用5折交叉验证  
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  
  
# 初始化变量来存储最佳验证损失和对应的模型  
best_val_loss = float('inf')  # 初始化为正无穷大  
best_model = None  
  
# 遍历KFold的每个分割  
for train_idx, val_idx in kfold.split(train_x):  
    # 分离训练集和验证集  
    train_x_fold, train_y_fold = train_x[train_idx], train_y[train_idx]  
    val_x_fold, val_y_fold = train_x[val_idx], train_y[val_idx]  
  
    # 创建和编译模型  
    model = Sequential()  
    model.add(LSTM(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))  
    model.add(Dropout(0.2))  
    model.add(Dense(64, activation='relu'))  
    model.add(Dropout(0.2))  
    model.add(Dense(32, activation='relu'))  
    model.add(Dense(1))  
      
    model.compile(loss='mse', optimizer='adam')  
  
    # 在当前折叠的数据上训练模型  
    history = model.fit(train_x_fold, train_y_fold, epochs=25, batch_size=32, validation_data=(val_x_fold, val_y_fold))  
      
    # 提取验证损失  
    val_loss = history.history['val_loss'][-1]  
      
    # 如果当前验证损失比之前保存的最佳验证损失要好，则更新最佳验证损失和模型  
    if val_loss < best_val_loss:  
        best_val_loss = val_loss  
        best_model = model  
      
# 打印最佳验证损失  
print('Best validation loss:', best_val_loss)  
model=best_model
def predict_future(model, initial_sequence, steps):
    predicted_values = []  # 存储预测结果
    current_sequence = initial_sequence.copy()  # 初始序列
    for i in range(steps):
        # 使用模型进行单步预测
        predicted_value = model.predict(current_sequence.reshape(1, initial_sequence.shape[0], initial_sequence.shape[1]))
        # 将预测结果添加到列表中
        predicted_values.append(predicted_value[0][0])
        # 更新当前序列，删除第一个时间步并将预测值添加到最后一个时间步
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = predicted_value
    return predicted_values
# 使用该函数进行预测
steps_to_predict =  12 # 要预测的步数
predicted_values = predict_future(model, test_x[-1], steps_to_predict)

train_max = np.max(df_train['商品房均价'])
train_min = np.min(df_train['商品房均价'])
series6 = np.array(predicted_values)*(train_max-train_min)+train_min
print(series6)


