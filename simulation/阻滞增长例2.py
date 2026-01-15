"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

def logistic_growth(t, N0, r, K, C, sigma):
    """
    阻滞增长模型函数

    参数:
        t: 时间
        N0: 初始数量
        r: 增长率
        K: 最大资源容量
        C: 自然消耗率
        sigma: 随机扰动的标准差

    返回值:
        随时间变化的资源数量
    """
    growth_rate = r * (1 - N0/K)
    noise = sigma * np.random.randn(len(t))
    return N0 / (1 + ((K / N0 - 1) * np.exp(-growth_rate * t))**(1/C)) + noise

# 参数设置
N0 = 1000  # 初始资源数量
r = 0.05  # 增长率
K = 5000  # 最大资源容量
C = 2  # 自然消耗率
sigma = 100  # 随机扰动的标准差

# 时间范围
t = np.linspace(0, 20, 200)

# 计算随时间变化的资源数量
N = logistic_growth(t, N0, r, K, C, sigma)

# 历史数据点
t_history = [2, 5, 10, 15]
N_history = [1200, 1800, 3500, 2000]

# 绘制资源消耗曲线和历史数据点
plt.plot(t, N, label='num', linewidth=2)
plt.scatter(t_history, N_history, color='red', label='history', marker='o', s=50)
plt.xlabel('time', fontsize=12)
plt.ylabel('num', fontsize=12)
plt.title('model', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()