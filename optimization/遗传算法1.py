"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import random
import string
import matplotlib.pyplot as plt

# 目标字符串，种群将进化以匹配这个字符串
target = "Hello, World!"
# 字母表
characters = string.ascii_letters + " ,.!?0123456789"


# 适应度函数，计算字符串与目标字符串的相似度
def fitness(guess):
    return sum(guess[i] == target[i] for i in range(len(target)))


# 随机生成一个初始种群中的个体
def create_individual(length):
    return ''.join(random.choice(characters) for _ in range(length))


# 交叉操作，交换两个个体的部分基因
def crossover(individual1, individual2):
    position = random.randint(0, len(target) - 1)
    return (individual1[:position] + individual2[position:],
            individual2[:position] + individual1[position:])


# 变异操作，随机修改个体的一个基因
def mutate(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        position = random.randint(0, len(target) - 1)
        individual = individual[:position] + random.choice(characters) + individual[position + 1:]
    return individual


# 遗传算法主流程
def genetic_algorithm():
    # 初始化
    generation = 0
    population_size = 100
    population = [create_individual(len(target)) for _ in range(population_size)]
    fitness_history = []

    while True:
        # 计算适应度并保存最佳个体
        population_fitness = [fitness(individual) for individual in population]
        max_fitness = max(population_fitness)
        best_individual = population[population_fitness.index(max_fitness)]
        fitness_history.append(max_fitness)

        # 可视化进化过程
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.pause(0.05)

        # 检查是否达到目标
        if best_individual == target:
            break

        # 选择过程（这里简化为取前50%最佳个体）
        sorted_pop = sorted(zip(population, population_fitness), key=lambda x: x[1], reverse=True)
        population = [individual for individual, fitness in sorted_pop[:population_size // 2]]

        # 用交叉和变异生成新个体，填充种群
        while len(population) < population_size:
            if random.random() > 0.1:  # 90% 的概率发生交叉
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring1, offspring2 = crossover(parent1, parent2)
                population.append(mutate(offspring1))
                population.append(mutate(offspring2))
            else:  # 10% 的概率添加全新个体
                population.append(create_individual(len(target)))

        # 更新代数
        generation += 1

    # 关闭可视化
    plt.close()
    return best_individual, generation


# 运行遗传算法
best_individual, generation = genetic_algorithm()
print(f"Generation: {generation}")
print(f"Best Individual: {best_individual}")