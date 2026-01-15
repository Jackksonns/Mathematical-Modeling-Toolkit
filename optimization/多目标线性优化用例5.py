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

# 找到最高点和最低点的位置
max_index = np.unravel_index(np.argmax(Z), Z.shape)
min_index = np.unravel_index(np.argmin(Z), Z.shape)

# 最高点坐标
max_x, max_y, max_z = X[max_index], Y[max_index], Z[max_index]
# 最低点坐标
min_x, min_y, min_z = X[min_index], Y[min_index], Z[min_index]

# 绘制地形图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')

# 绘制最高点竖线
ax.plot([max_x, max_x], [max_y, max_y], [0, max_z], color='red')

# 绘制最低点竖线
ax.plot([min_x, min_x], [min_y, min_y], [0, min_z], color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Terrain Modeling')

plt.show()