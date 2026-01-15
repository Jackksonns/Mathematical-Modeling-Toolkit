"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the function to be optimized
def func_to_optimize(x):
    return (x - 3) ** 2


# Monte Carlo Optimization
def monte_carlo_optimization(iterations):
    best_x = None
    best_y = float('inf')
    x_values = []
    y_values = []

    for i in range(iterations):
        x = np.random.uniform(0, 6)  # Generate a random x value in the range [0, 6]
        y = func_to_optimize(x)
        x_values.append(x)
        y_values.append(y)

        # If this is the best x seen so far, record it
        if y < best_y:
            best_x = x
            best_y = y

    return best_x, best_y, x_values, y_values


# Run the optimization
iterations = 1000
best_x, best_y, x_values, y_values = monte_carlo_optimization(iterations)

# Print the results
print(f"Optimal x: {best_x}")
print(f"Optimal y: {best_y}")

# Visualization
plt.scatter(x_values, y_values, color='blue', marker='o', label='Sample Points')
plt.scatter(best_x, best_y, color='red', marker='x', label='Optimal Point')
plt.axvline(x=3, color='green', linestyle='--', label='x=3 (True Minima)')
plt.title('Monte Carlo Optimization Results')
plt.xlabel('x value')
plt.ylabel('f(x)')
plt.legend()
plt.show()