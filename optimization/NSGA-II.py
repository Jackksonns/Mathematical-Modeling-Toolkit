"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt

# 定义优化问题的目标数量和变量数量  
n_objectives = 2
n_variables = 2

# 创建问题类型  
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # weights定义了优化目标的优先级  
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)  # 定义变量范围  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_variables)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalFunc(individual):  # 定义目标函数
    x1, x2 = individual
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 1) ** 2 + x2 ** 2
    return f1, f2


toolbox.register("evaluate", evalFunc)
toolbox.register("mate", tools.cxTwoPoint)  # 定义交叉操作  
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # 定义变异操作  
toolbox.register("select", tools.selNSGA2)  # 选择操作采用NSGA-II的选择策略  


def main():
    pop = toolbox.population(n=100)  # 定义种群大小  
    hof = tools.HallOfFame(1)  # 创建一个名人堂，用于存放历代最优个体  
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # 定义统计对象，用于收集优化过程中的信息  
    stats.register("avg", np.mean, axis=0)  # 注册统计信息，计算每代种群的平均适应度  
    stats.register("std", np.std, axis=0)  # 注册统计信息，计算每代种群的适应度标准差  
    stats.register("min", np.min, axis=0)  # 注册统计信息，找出每代种群的最小适应度  
    stats.register("max", np.max, axis=0)  # 注册统计信息，找出每代种群的最大适应度  
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof,
                                       verbose=True)  # 运行算法
    return pop, logbook, hof


if __name__ == "__main__":
    pop, logbook, hof = main()
    print("Best individual: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))  # 输出最优个体和对应的适应度值

import matplotlib.pyplot as plt


def plot_results(population):
    # 提取目标函数值
    objectives = [ind.fitness.values for ind in population]
    f1_values = [obj[0] for obj in objectives]
    f2_values = [obj[1] for obj in objectives]

    # 绘制散点图
    plt.scatter(f1_values, f2_values)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Scatter Plot of ZDT2 Results')
    plt.show()

    # 绘制Pareto前沿图
    plt.scatter(f1_values, f2_values)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Pareto Front of ZDT2 Results')
    plt.xlim([0, 1])  # 设置x轴范围
    plt.ylim([0, 1])  # 设置y轴范围
    plt.plot([0, 1], [0, 1], 'r--')  # 绘制真实的Pareto前沿
    plt.show()