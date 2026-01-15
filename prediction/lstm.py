"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 生成0到99之间的数字作为数据
data = np.array([[i] for i in range(100)]).astype(np.float32)

# 生成目标值，这次我们使用一个简单的线性方程 y = 3x + 5 来生成目标值
target = 3 * data* data + 5* data + 4
data = np.reshape(data, (100, 1, 1))
target = np.reshape(target, (100, 1))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(10, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(data, target, epochs=1000, batch_size=1, verbose=2)

# 进行预测
test_data = np.array([[i] for i in range(101, 201)])
test_data = np.reshape(test_data, (100, 1, 1))
predictions = model.predict(test_data)

# 可视化结果
plt.plot(range(1, 202), [i * 2 for i in range(1, 202)], label='True')
plt.plot(range(1, 101), predictions, label='Predicted')
plt.legend()
plt.show()
