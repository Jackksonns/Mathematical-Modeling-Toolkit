"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取灰度图像
gray_image = plt.imread(r'C:\Users\Administrator\Desktop\MathCode\A\海底地形示例图3.png')

# 转换为单通道的灰度图像
gray_image = gray_image[:, :, 0]

# 获取图像尺寸
height, width = gray_image.shape

# 构建坐标网格
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)

# 构建地形数据
Z = gray_image

# 计算梯度
gradient_x = np.gradient(Z, axis=1)
gradient_y = np.gradient(Z, axis=0)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# 找到梯度变化最大的区域
max_gradient_region = gradient_magnitude > np.mean(gradient_magnitude)

# 绘制地形图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')

# 绘制梯度变化最大的区域
ax.contour(X, Y, max_gradient_region, colors='r', levels=[0.5])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Terrain Modeling')

plt.show()