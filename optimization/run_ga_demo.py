"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""


import numpy as np
import matplotlib.pyplot as plt
from imyl.optimization.genetic_algorithm import GeneticAlgorithm

# Define objective function
def objective_function(x):
    return -(x ** 2) + 4 * x + 3

def main():
    print("Running Genetic Algorithm Demo...")
    
    # Configuration
    bounds = (-10, 10)
    optimizer = GeneticAlgorithm(
        objective_function=objective_function,
        bounds=bounds,
        population_size=50,
        num_generations=100,
        mutation_rate=0.1
    )
    
    # Run Optimization
    best_x, best_y = optimizer.fit()
    
    print(f"Optimization Complete.")
    print(f"Best Solution x: {best_x:.4f}")
    print(f"Best Fitness y: {best_y:.4f}")
    
    # Visualization
    x_values = np.linspace(bounds[0], bounds[1], 1000)
    y_values = objective_function(x_values)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Objective Function')
    
    final_pop = optimizer.population
    final_pop_fitness = objective_function(final_pop)
    
    plt.scatter(final_pop, final_pop_fitness, color='red', marker='x', label='Final Population')
    plt.scatter([best_x], [best_y], color='green', s=100, label='Best Solution')
    
    plt.xlabel('x')
    plt.ylabel('Objective Function Value')
    plt.title('Genetic Algorithm Optimization (Modularized)')
    plt.legend()
    plt.grid(True)
    
    # Save figure instead of showing it to avoid blocking
    output_path = 'ga_demo_result.png'
    plt.savefig(output_path)
    print(f"Result plot saved to {output_path}")

if __name__ == "__main__":
    main()
