"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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

# 城市A的人口数据
population_A = [100, 150, 200, 250, 300, 350, 400, 450, 500]  # 历史人口数据
years_A = range(1, len(population_A) + 1)  # 对应的年份

# 拟合阻滞增长模型
popt, pcov = scipy.optimize.curve_fit(logistic_growth, years_A, population_A)

# 预测未来10年的人口增长
years_future = range(1, len(population_A) + 11)  # 未来10年的年份
population_future = logistic_growth(years_future, *popt)

# 绘制人口增长曲线
plt.plot(years_A, population_A, 'bo', label='histoy')
plt.plot(years_future, population_future, 'r-', label='p-pred')
plt.xlabel('year')
plt.ylabel('num')
plt.title('city-p-pred')
plt.legend()
plt.show()