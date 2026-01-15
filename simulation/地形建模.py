"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_mountain_from_terrain(image_path):
    terrain_image = plt.imread(image_path)
    height_map = terrain_image.mean(axis=-1) * 255

    x_size, y_size = height_map.shape
    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    x, y = np.meshgrid(x, y)

    # Adjust the scaling factor as needed
    scaling_factor = 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001




    z = scaling_factor * height_map

    return x, y, z

def main():
    terrain_image_path = r'C:\Users\Administrator\Desktop\MathCode\2-优化-2\地形灰度图.jpg'

    x, y, z = generate_mountain_from_terrain(terrain_image_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.T, cmap=cm.terrain, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



if __name__ == "__main__":
    main()
