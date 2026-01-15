"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 定义目标函数
def objectives(x):
    x1, x2 = x
    f1 = x1**2 + x2**2
    f2 = (x1 - 5)**2 + (x2 - 5)**2
    return f1 + 0.5 * f2


# 定义初始猜测值
x0 = [0, 0]

# 使用 scipy 的 minimize 函数进行优化
result = minimize(objectives, x0, method='Nelder-Mead')

# 可视化结果
x1, x2 = result.x
plt.scatter([x1], [x2], color='red', label='Pareto front')
plt.scatter([0, 10], [0, 10], color='blue', label='f1')
plt.scatter([5, 5], [0, 10], color='green', label='f2')
plt.legend()
plt.show()