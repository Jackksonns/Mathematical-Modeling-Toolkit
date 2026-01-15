"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

def dtlz2(x, M):
    N = len(x)
    g = np.sum((x[M-1:] - 0.5)**2)
    f = np.zeros(M)
    for i in range(M-1):
        f[i] = (1 + g) * np.prod(np.cos(x[:i] * np.pi/2))
    f[M-1] = (1 + g) * np.prod(np.cos(x[:M-1] * np.pi/2)) * np.sin(x[M-1] * np.pi/2)
    return f

# 定义多目标规划模型
def multi_objective_optimization(problem, num_variables, lower_bounds, upper_bounds, num_points, M):
    variables = np.random.uniform(lower_bounds, upper_bounds, (num_points, num_variables))
    objectives = np.zeros((num_points, M))
    for i in range(num_points):
        objectives[i] = problem(variables[i], M)
    return variables, objectives

# 设置决策变量的范围、数量和目标函数的数量
num_variables = 10
lower_bounds = np.zeros(num_variables)
upper_bounds = np.ones(num_variables)
num_points = 1000
M = 3

# 解决DTLZ2问题并获取决策变量和目标函数值
variables, objectives = multi_objective_optimization(dtlz2, num_variables, lower_bounds, upper_bounds, num_points, M)

# 计算Pareto前沿
def pareto_front(objectives):
    sorted_indices = np.argsort(objectives[:, 0])
    pareto_front = [sorted_indices[0]]
    for index in sorted_indices[1:]:
        if objectives[index, 1] < objectives[pareto_front[-1], 1]:
            pareto_front.append(index)
    return pareto_front

pareto_indices = pareto_front(objectives)
pareto_front = objectives[pareto_indices]

# 可视化输出Pareto前沿图
plt.scatter(objectives[:, 0], objectives[:, 1], c='b', label='Non-Dominated Solutions')
plt.plot(pareto_front[:, 0], pareto_front[:, 1], c='r', label='Pareto Front')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('DTLZ2 Problem')
plt.legend()
plt.show()

#运行以上代码，将会得到一个散点图，并绘制了Pareto前沿图。
# 其中 x 轴表示目标函数1的值，y 轴表示目标函数2的值。
# 图中的散点代表了多个解，红色曲线表示Pareto前沿。
# 这样，你就可以直观地观察到解的分布情况以及Pareto前沿的形状。
# 你可以根据需要调整问题的维度、决策变量的范围、数量和目标函数的数量，以及优化算法的参数来进一步探索和优化多目标问题。
#rzna@foxmai.com ruan-美赛保奖班专属