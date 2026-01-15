"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 遗传算法参数设置
population_size = 100  # 种群大小
num_generations = 100  # 迭代次数
mutation_rate = 0.02  # 变异率

# 城市坐标
cities = np.array([[2, 3], [7, 8], [1, 6], [4, 4], [9, 2],
                   [5, 1], [3, 7], [6, 5], [8, 4], [1, 9],
                   [2, 2], [6, 9], [4, 7], [9, 1], [3, 3],
                   [7, 6], [5, 8], [8, 2], [1, 4], [9, 7]])

def calculate_total_distance(path):
    """
    计算路径的总距离

    参数:
        path: 路径顺序

    返回值:
        路径的总距离
    """
    distances = cdist(cities[path[:-1]], cities[path[1:]], 'euclidean')
    return np.sum(distances)

def generate_initial_population(size, num_cities):
    """
    生成初始种群

    参数:
        size: 种群大小
        num_cities: 城市数量

    返回值:
        初始种群
    """
    population = []
    for _ in range(size):
        individual = np.random.permutation(num_cities)
        population.append(individual)
    return population

def mutate_individual(individual, mutation_rate):
    """
    变异个体

    参数:
        individual: 个体
        mutation_rate: 变异率

    返回值:
        变异后的个体
    """
    if np.random.rand() < mutation_rate:
        idx = np.random.choice(len(individual), size=2, replace=False)
        individual[idx[0]], individual[idx[1]] = individual[idx[1]], individual[idx[0]]
    return individual

def crossover(parent1, parent2):
    """
    交叉生成子代

    参数:
        parent1: 父代1
        parent2: 父代2

    返回值:
        子代
    """
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.hstack((parent1[:crossover_point], np.setdiff1d(parent2, parent1[:crossover_point])))
    child2 = np.hstack((parent2[:crossover_point], np.setdiff1d(parent1, parent2[:crossover_point])))
    return child1, child2

def select_parents(population):
    """
    选择父代

    参数:
        population: 种群

    返回值:
        父代1, 父代2
    """
    fitness = [1 / calculate_total_distance(individual) for individual in population]
    fitness_probs = np.array(fitness) / np.sum(fitness)
    parents_idx = np.random.choice(len(population), size=2, replace=False, p=fitness_probs)
    return population[parents_idx[0]], population[parents_idx[1]]

def optimize_path(cities, population_size, num_generations, mutation_rate):
    """
    优化路径

    参数:
        cities: 城市坐标
        population_size: 种群大小
        num_generations: 迭代次数
        mutation_rate: 变异率

    返回值:
        最优路径
    """
    num_cities = len(cities)
    population = generate_initial_population(population_size, num_cities)

    best_distance = float('inf')
    best_path = None

    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population

        # 计算当前最优路径
        for individual in population:
            distance = calculate_total_distance(individual)
            if distance < best_distance:
                best_distance = distance
                best_path = individual

    return best_path

# 优化路径
best_path = optimize_path(cities, population_size, num_generations, mutation_rate)

# 绘制城市和路径
plt.figure(figsize=(8, 6))
plt.scatter(cities[:, 0], cities[:, 1], color='blue', s=100, label='city')
plt.plot(cities[best_path][:, 0], cities[best_path][:, 1], color='red', linewidth=2, label='best-track')  # 修改这一行
plt.xlabel('X ', fontsize=12)
plt.ylabel('Y ', fontsize=12)
plt.title('modle', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()