"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

def logistic_growth(t, N0, r, K):
    """
    阻滞增长模型函数

    参数:
        t: 时间
        N0: 初始数量
        r: 增长率
        K: 稳定数量

    返回值:
        随时间变化的数量
    """
    return K / (1 + (K / N0 - 1) * np.exp(-r * t))

# 参数设置
N0 = 100  # 初始数量
r = 0.1  # 增长率
K = 1000  # 稳定数量

# 时间范围
t = np.linspace(0, 10, 100)

# 计算随时间变化的数量
N = logistic_growth(t, N0, r, K)

# 绘制增长曲线

plt.plot(t, N)
plt.xlabel('time')
plt.ylabel('num')
plt.title('model')

plt.show()