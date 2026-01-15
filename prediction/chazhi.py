"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(xi, yi, x):
    """
    对给定的数据点进行拉格朗日插值
    参数:
        xi: np.array, 数据点的x坐标
        yi: np.array, 数据点的y坐标
        x: 插值点的x坐标
    返回:
        y: 插值点的y坐标
    """
    n = len(xi)
    y = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (x - xi[j]) / (xi[i] - xi[j])
        y += p * yi[i]
    return y


# 测试代码
xi = np.array([0, 1, 2, 3])
yi = np.array([-1, 0, 7, 26])
x = np.linspace(min(xi), max(xi), 1000)  # 生成1000个插值点的x坐标
y = lagrange_interpolation(xi, yi, x)  # 计算插值点的y坐标

# 绘制散点图和插值曲线
plt.scatter(xi, yi, color='red', label='chazhi')
plt.plot(x, y, color='blue', label='nihe')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Lagrange Interpolation')
plt.grid(True)
plt.show()