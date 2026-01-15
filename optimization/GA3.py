"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义适应度函数，用于评估个体的优劣
def fitness_function(individual):
    # 这是一个简单的适应度函数，根据问题的具体需求可以进行修改
    # 这里将个体的三个变量的平方和作为适应度值
    return np.sum(np.square(individual))

# 设置遗传算法的参数
population_size = 50      # 种群大小
num_generations = 50      # 进化代数
mutation_rate = 0.1       # 变异率
crossover_rate = 0.8      # 交叉率

# 初始化种群，生成一个大小为 (population_size, 3) 的随机矩阵，每个个体的取值范围在 [0, 10)
population = np.random.rand(population_size, 3) * 10

# 开始遗传算法的主循环
for generation in range(num_generations):
    # 计算种群中每个个体的适应度值
    fitness_values = np.apply_along_axis(fitness_function, 1, population)

    # 选择操作，使用轮盘赌选择法选择个体，适应度值越高的个体被选中的概率越大
    selected_indices = np.random.choice(range(population_size), size=population_size, p=fitness_values / np.sum(fitness_values))
    # 根据选择的索引，将选中的个体放入新的种群中（这里没有显示地写出新种群的创建，但实际上selected_indices记录了被选中个体的索引）

    # 交叉操作，以一定的交叉率对选中的个体进行交叉操作，生成新的个体
    crossover_indices = np.random.choice(range(population_size), size=int(crossover_rate * population_size), replace=False)
    for idx in crossover_indices:
        # 为当前个体选择两个交叉伙伴
        partner_indices = np.random.choice(range(population_size), size=2, replace=False)
        # 随机选择一个交叉点，将当前个体与两个交叉伙伴在交叉点后的基因进行交换
        crossover_point1 = np.random.randint(3)
        crossover_point2 = np.random.randint(3)
        population[idx, crossover_point1:], population[partner_indices[0], crossover_point1:] = (
            population[partner_indices[0], crossover_point1:], population[idx, crossover_point1:]
        )
        population[idx, crossover_point2:], population[partner_indices[1], crossover_point2:] = (
            population[partner_indices[1], crossover_point2:], population[idx, crossover_point2:]
        )

    # 变异操作，以一定的变异率对个体进行变异操作，增加种群的多样性
    mutation_indices = np.random.choice(range(population_size), size=int(mutation_rate * population_size), replace=False)
    # 对选中的个体进行变异，这里使用正态分布的随机噪声作为变异量
    population[mutation_indices] += np.random.randn(len(mutation_indices), 3)

# 计算最终种群的适应度值，并找出最优个体
final_fitness = np.apply_along_axis(fitness_function, 1, population)
best_individual = population[np.argmin(final_fitness)]

# 使用3D散点图可视化种群的分布情况，点的颜色表示适应度值的大小
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(population[:, 0], population[:, 1], population[:, 2], c=final_fitness, cmap='viridis')
# 使用红色'x'标记最优个体的位置
ax.scatter(best_individual[0], best_individual[1], best_individual[2], marker='x', color='red', label='Best Individual')
# 显示颜色条，表示适应度值的大小
fig.colorbar(scatter, label='Fitness')
ax.set_xlabel('Variable 1')  # 设置x轴标签
ax.set_ylabel('Variable 2')  # 设置y轴标签
ax.set_zlabel('Variable 3')  # 设置z轴标签
ax.set_title('Population Optimization with Genetic Algorithm')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表
