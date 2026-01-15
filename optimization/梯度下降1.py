"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义目标函数和梯度函数
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

def grad_f(x, y):
    r = np.sqrt(x**2 + y**2)
    grad_x = x / r * np.cos(r)
    grad_y = y / r * np.cos(r)
    return -grad_x, -grad_y  # 返回负梯度，因为我们要进行下降

# 梯度下降算法
def gradient_descent(grad, start_x, start_y, learning_rate, num_iterations):
    x, y = start_x, start_y
    x_path, y_path = [], []
    for _ in range(num_iterations):
        dx, dy = grad(x, y)
        x, y = x + learning_rate * dx, y + learning_rate * dy
        x_path.append(x)
        y_path.append(y)
    return x_path, y_path

# 初始化参数
start_x, start_y = 1.0, 1.0  # 初始点
learning_rate = 0.1  # 学习率
num_iterations = 50  # 迭代次数

# 进行梯度下降优化
x_path, y_path = gradient_descent(grad_f, start_x, start_y, learning_rate, num_iterations)

# 可视化
fig = plt.figure(figsize=(10, 6))
# 使用 subplot 创建3D轴
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(-2, 2, 400)
Y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# 画出目标函数的曲面
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.8)

# 画出梯度下降的路径
ax.plot(x_path, y_path, f(np.array(x_path), np.array(y_path)), color='r', marker='*', markersize=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Gradient Descent Optimization and Path')

plt.show()