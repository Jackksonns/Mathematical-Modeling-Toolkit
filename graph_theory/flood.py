"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

def flood_fill(maze, start_x, start_y):
    rows, cols = maze.shape
    queue = [(start_x, start_y)]
    visited = np.zeros_like(maze)

    while queue:
        x, y = queue.pop(0)
        visited[x, y] = 1

        if maze[x, y] == 2:  # Found the exit
            return visited

        # Check neighboring cells
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] != 1 and visited[nx, ny] == 0:
                queue.append((nx, ny))
                visited[nx, ny] = 1

    return visited

# Define the maze
maze = np.array([
    [0, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1]
])

# Perform flood fill to find the exit
visited = flood_fill(maze, 0, 0)

# Create a colormap for visualization
cmap = colors.ListedColormap(['white', 'blue', 'red', 'green'])
bounds = [0, 1, 2, 3, 4]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Create figure and axis objects
fig, ax = plt.subplots()

# Function to update the plot for each frame of the animation
def update(frame):
    ax.clear()
    ax.imshow(visited[:frame+1], cmap=cmap, norm=norm)

    # Add gridlines
    ax.grid(which='both', color='grey', linewidth=1)
    ax.set_xticks(np.arange(-0.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(maze), 1), minor=True)
    ax.tick_params(which='minor', length=0)

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=len(visited), interval=500, repeat=False)

plt.show()