"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""


import numpy as np
import matplotlib.pyplot as plt

# 参数设置
lambda_ = 0.015  # 人口增长率
N0 = 1000000  # 初始人口数量
T = 10  # 模拟时间长度（年）

# 模拟人口变化
population = N0
events = np.random.poisson(lambda_ * population, T)
population += events

# 绘制人口数量随时间变化的图像
plt.plot(population)
plt.xlabel('Time (years)')
plt.ylabel('Population size')
plt.title('Random population model simulation')
plt.show()
