"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter


# 适应度函数
def fitness_function(individual):
    x, y = individual
    return x ** 2 + y ** 2


# 遗传算法参数设置
population_size = 50  # 种群大小
num_generations = 50  # 进化代数
mutation_rate = 0.1  # 变异率
crossover_rate = 0.8  # 交叉率

# 初始化种群
population = np.random.rand(population_size, 2) * 10 - 5  # 初始值在 [-5, 5] 范围内

# 开始遗传算法主循环
for generation in range(num_generations):
    # 计算适应度值
    fitness_values = np.apply_along_axis(fitness_function, 1, population)

    # 选择操作（轮盘赌选择法）
    selected_indices = np.random.choice(range(population_size), size=population_size,
                                        p=fitness_values / np.sum(fitness_values))
    population = population[selected_indices]

    # 交叉操作（单点交叉）
    crossover_indices = np.random.choice(range(population_size), size=int(crossover_rate * population_size),
                                         replace=False)
    for idx in crossover_indices:
        partner_idx = np.random.choice(range(population_size))
        crossover_point = np.random.randint(2)
        if crossover_point == 0:
            population[idx, :] = population[partner_idx, :]
        else:
            population[idx, crossover_point] = population[partner_idx, crossover_point]

            # 变异操作（均匀变异）
    mutation_indices = np.random.choice(range(population_size), size=int(mutation_rate * population_size),
                                        replace=False)
    population[mutation_indices] += np.random.uniform(-1, 1, size=(len(mutation_indices), 2))  # 在 [-1, 1] 范围内变异

# 找到最优个体并计算其适应度值
best_individual = population[np.argmin(fitness_values)]
best_fitness = fitness_function(best_individual)
print(f"Best individual: {best_individual}, Fitness: {best_fitness}")

# 可视化种群分布情况（使用matplotlib）和最优解的位置（使用红色'x'标记）
fig, ax = plt.subplots()
ax.scatter(population[:, 0], population[:, 1], c=fitness_values, cmap='viridis')
ax.scatter(best_individual[0], best_individual[1], c='red', marker='x', s=100)  # 使用红色'x'标记最优个体的位置
ax2 = ax.twinx()  # 创建第二个y轴用于适应度表示
ax2.set_ylabel('Fitness')  # 设置第二个y轴的标签为适应度
ax2.set_ylim([0, max(fitness_values)])  # 设置第二个y轴的范围为[0, max fitness]
ax2.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x:.2f}"))  # 设置刻度标签格式为保留两位小数，例如显示为 "0.99" 而不是 "0.9934" 等。
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_title('Genetic Algorithm Evolution')
plt.show()