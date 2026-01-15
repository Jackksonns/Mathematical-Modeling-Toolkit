"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x)-1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return f1, f2

# 定义多目标规划模型
def multi_objective_optimization(problem, num_variables, lower_bounds, upper_bounds, num_points):
    variables = np.random.uniform(lower_bounds, upper_bounds, (num_points, num_variables))
    objectives = np.zeros((num_points, 2))
    for i in range(num_points):
        objectives[i] = problem(variables[i])
    return variables, objectives

# 设置决策变量的范围和数量
num_variables = 30
lower_bounds = [0] + [-5] * (num_variables - 1)
upper_bounds = [1] + [5] * (num_variables - 1)

# 解决ZDT1问题并获取决策变量和目标函数值
num_points = 1000
variables, objectives = multi_objective_optimization(zdt1, num_variables, lower_bounds, upper_bounds, num_points)

# 可视化输出
plt.scatter(objectives[:, 0], objectives[:, 1], c='b', label='ZDT1')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('ZDT1 Problem')
plt.legend()
plt.show()

#将会得到一个散点图，其中 x 轴表示目标函数1的值，y 轴表示目标函数2的值。图中的散点代表了多个解，并展示了它们之间的关系
#rzna@foxmail.com ruan-美赛保奖班专属
