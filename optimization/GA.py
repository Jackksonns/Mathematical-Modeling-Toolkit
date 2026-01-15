"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数（此处以一个简单的二次函数为例）
def objective_function(x):
    return -(x ** 2) + 4 * x + 3


# 遗传算法的参数
population_size = 50  # 种群大小  
num_generations = 100  # 迭代次数  
mutation_rate = 0.1  # 变异率  

# 初始化种群，这里随机生成一个包含50个数的数组，范围在-10到10之间  
population = np.random.uniform(low=-10, high=10, size=population_size)

# 主循环，进行100次迭代  
for generation in range(num_generations):
    # 计算种群中每个个体的适应度（即目标函数的值）  
    fitness = objective_function(population)

    # 根据适应度选择父代，这里选择适应度最高的前一半个体作为父代  
    selected_indices = np.argsort(fitness)[-population_size // 2:]
    parents = population[selected_indices]

    # 进行交叉操作（单点交叉），随机选择一个点作为交叉点，然后分段形成新的后代  
    crossover_point = np.random.randint(1, population_size // 2)
    children = np.concatenate([parents[:crossover_point], parents[crossover_point:]])

    # 进行变异操作，随机选择一部分后代进行小幅度扰动，模拟基因突变  
    mutation_mask = (np.random.rand(*children.shape) < mutation_rate)
    children += mutation_mask * np.random.normal(0, 1, size=children.shape)

    # 用新的后代替换当前种群的前半部分和后半部分  
    population[:population_size // 2] = parents
    population[population_size // 2:] = children

# 可视化结果，先画出目标函数，然后画出种群的位置（用红色叉号表示）  
x_values = np.linspace(-10, 10, 1000)  # 在-10到10之间生成1000个点，用于绘图  
y_values = objective_function(x_values)  # 计算这些点的目标函数值  

plt.plot(x_values, y_values, label='Objective Function')  # 画出目标函数  
plt.scatter(population, objective_function(population), color='red', marker='x', label='Population')  # 用红色叉号表示种群的位置  
plt.xlabel('x')  # x轴标签  
plt.ylabel('Objective Function Value')  # y轴标签 ：目标函数值
plt.legend()  # 显示图例  
plt.title('Genetic Algorithm Optimization')  # 图表标题 ：遗传算法优化
plt.show()  # 显示图表