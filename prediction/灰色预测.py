"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def GM11(x0):
    # 累加生成序列
    x1 = x0.cumsum()
    # 紧邻均值生成序列
    z1 = (x1[: -1] + x1[1:]) / 2.0
    n = len(x0)
    # 数据矩阵B
    B = np.append(-z1.reshape((n - 1, 1)), np.ones((n - 1, 1)), axis=1)
    # 数据向量Yn
    Yn = x0[1:].reshape((n - 1, 1))
    # 计算参数a, u
    [[a], [u]] = np.linalg.inv(B.T @ B) @ B.T @ Yn

    # 建立GM(1,1)模型
    def model(k):
        return (x0[0] - u / a) * np.exp(-a * (k - 1)) - (x0[0] - u / a) * np.exp(-a * (k - 2))

    return model, a, u


def predict(x0, model, n):
    # 预测序列
    predict = [model(i + 1) for i in range(n)]
    return np.array(predict)


def evaluate(x0, predict):
    # 残差
    residual = x0 - predict
    # 相对误差
    relative_error = residual / x0
    return residual, relative_error


# 示例数据
data = pd.Series([3, 5, 8, 13, 21, 34])

# 建立模型
model, a, u = GM11(data.values)

# 预测
n = len(data)
predict = predict(data.values, model, n)

# 评估
residual, relative_error = evaluate(data.values, predict)

# 输出结果
print("Predicted values:", predict)
print("Residuals:", residual)
print("Relative errors:", relative_error)

# 可视化结果
plt.figure(figsize=(8, 4))
plt.plot(data.index, data.values, 'o-', label='Actual')
plt.plot(data.index, predict, 'r--', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Gray Prediction Model')
plt.legend()
plt.show()