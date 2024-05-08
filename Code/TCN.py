import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## keras
from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Attention
from tcn import TCN

df = pd.read_csv(r'D:\MASTER\my_paper\paper_2\Processed data\Perturbation Testing\Chu.csv')
values = df.values
print(values.shape)
# 将序列转为监督学习序列数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 丢弃nan值
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 提取特征列和目标变量列
# features = values[:, :-1]
# target = values[:, -1]
# # 归一化特征列
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_features = scaler.fit_transform(features)
#
# # 连接归一化后的特征列和目标变量列
# scaled_values = np.column_stack((scaled_features, target))
# 将数据缩放到0到1之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)
print(scaled_values.shape)
#print(scaled_values.shape)
# 数据转为监督学习数据
reframed = series_to_supervised(scaled_values, 1, 1)

# 划分数据集
values = reframed.values
print(values.shape)

# 划分数据集为训练集、验证集和测试集 (6:2:2)
n_total = len(values)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.2)

train = values[:n_train, :]
val = values[n_train:n_train + n_val, :]
test = values[n_train + n_val:, :]
print(train.shape, val.shape, test.shape)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 构建数据的3D 格式，即[samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)


model = Sequential()
model.add(TCN(nb_filters=32,kernel_size=3,padding='causal', dilations=(1, 2, 4, 8,16), input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=64,
                    validation_data=(val_X, val_y), verbose=2, shuffle=False)

# # 显示训练的loss值情况
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()


# 做预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 21))

# i反归一化预测值
inv_yhat = np.concatenate((test_X[:, -10:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
# 反归一化真实值
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, -10:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# 将预测值输出为csv文件
# import csv
# # 要显示的值
# values = inv_yhat
# # CSV文件路径
# csv_file = r'D:\MASTER\my_paper\paper_2\Result\Perturbation Testing\TCN\Chu10'
# # 将值显示在CSV文件的一列中
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入列标题
#     writer.writerow(['Pre'])
#     # 写入值到一列
#     for value in values:
#         writer.writerow([value])

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# 计算平均绝对误差MAE
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)
# 计算平均绝对百分比误差MAPE
mape = (np.abs(inv_yhat - inv_y) / np.abs(inv_y)).mean()
print('Test MAPE: %.3f' % mape)
# 计算R2
print('r2_score: %.3f' % r2_score(inv_y, inv_yhat))
mean_observed = np.mean(inv_y)
# 计算RRMSE
rrmse = rmse / mean_observed
print('RRMSE: %.3f' % rrmse)
# 计算RMAE
rmae = mae / mean_observed
print('RMAE: %.3f' % rmae)
# 计算KGE
r = np.corrcoef(inv_y, inv_yhat)[0,1]  # 相关系数
# 计算标准差
std_observed = np.std(inv_y)
std_simulated = np.std(inv_yhat)
mean_simulated = np.mean(inv_yhat)
kge = 1 - np.sqrt((r - 1)**2 + (mean_simulated/mean_observed - 1)**2 + (std_simulated/std_observed - 1)**2)
print('KGE: %.3f' % kge)


# 为了改进模型，必须调整epoch和Batch_size等超参数

# #显示预测结果
# aa = [x for x in range(len(inv_yhat))]
# plt.plot(aa, inv_y, marker='.', label="actual")
# plt.plot(aa, inv_yhat, 'r', label="prediction")
# plt.ylabel('runoff', size=15)
# plt.xlabel('Time', size=15)
# plt.legend(fontsize=15)
# plt.show()