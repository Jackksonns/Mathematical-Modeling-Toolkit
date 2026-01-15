"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取灰度图像
gray_image = plt.imread(r"C:\Users\Administrator\Desktop\9.上课配套代码\2-优化-2\地形灰度图.jpg")

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

# 计算平均高度
average_height = np.mean(Z)

# 绘制地形图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')

# 绘制浅红色平面
ax.plot_surface(X, Y, np.full_like(Z, average_height), color='lightcoral', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Terrain Modeling')

plt.show()