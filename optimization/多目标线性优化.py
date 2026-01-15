"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 定义目标函数
def f1(x, y):
    return x**2 + y**2

def f2(x, y):
    return (x - 5)**2 + (y - 5)**2

# 定义目标函数的总和
def objectives(x):
    x1, x2 = x
    return f1(x1, x2) + f2(x1, x2)

# 生成网格点
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# 计算目标函数值
Z1 = f1(X, Y)
Z2 = f2(X, Y)

# 绘制三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)

# 定义目标函数
def objectives(x):
    x1, x2 = x
    f1 = x1**2 + x2**2
    f2 = (x1 - 5)**2 + (x2 - 5)**2
    return f1 + f2  # 返回目标函数的和

# 定义初始猜测值
x0 = [0, 0]

# 使用scipy的minimize函数进行优化
result = minimize(objectives, x0, method='Nelder-Mead')

# 绘制最优解点
x1, x2 = result.x
ax.scatter([x1], [x2], [objectives([x1, x2])], color='red', s=50, label='Optimal Point')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Objective Functions')
ax.legend()
plt.show()