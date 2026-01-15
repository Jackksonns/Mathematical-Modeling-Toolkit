"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

# 随机生成三段平滑的曲线
np.random.seed(42)

# 曲线1
x1 = np.linspace(0, 10, 100)
y1 = np.random.uniform(0, 5, size=100)
tck1, u1 = splprep([x1, y1], s=0)
x1_smooth, y1_smooth = splev(np.linspace(0, 1, 300), tck1)

# 曲线2
x2 = np.linspace(5, 15, 120)
y2 = np.random.uniform(2, 8, size=120)
tck2, u2 = splprep([x2, y2], s=0)
x2_smooth, y2_smooth = splev(np.linspace(0, 1, 300), tck2)

# 曲线3
x3 = np.linspace(10, 20, 80)
y3 = np.random.uniform(0, 6, size=80)
tck3, u3 = splprep([x3, y3], s=0)
x3_smooth, y3_smooth = splev(np.linspace(0, 1, 300), tck3)

# 生成随机点
random_point = np.random.uniform(2, 18, size=(2,))

# 计算点到曲线的距离
distances = cdist(np.array([random_point]), np.vstack((np.column_stack((x1_smooth, y1_smooth)), np.column_stack((x2_smooth, y2_smooth)), np.column_stack((x3_smooth, y3_smooth)))), 'euclidean')
closest_index = np.argmin(distances)

# 绘制曲线和随机点
plt.figure(figsize=(8, 6))
plt.plot(x1_smooth, y1_smooth, 'r-', label='Curve 1')
plt.plot(x2_smooth, y2_smooth, 'g-', label='Curve 2')
plt.plot(x3_smooth, y3_smooth, 'b-', label='Curve 3')
plt.plot(random_point[0], random_point[1], 'ko', label='Random Point')

# 绘制到达绿色曲线的直线距离
closest_x, closest_y = np.vstack((np.column_stack((x1_smooth, y1_smooth)), np.column_stack((x2_smooth, y2_smooth)), np.column_stack((x3_smooth, y3_smooth))))[closest_index]
plt.plot([random_point[0], closest_x], [random_point[1], closest_y], 'm--', label='Closest Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Curves and Closest Line')
plt.legend()
plt.grid(True)
plt.show()