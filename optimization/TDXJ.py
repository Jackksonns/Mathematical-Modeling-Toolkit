"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np

# 定义交通网络的节点和道路
num_nodes = 5
num_roads = 8
roads = np.array([
    [0, 1, 100],  # 道路0连接节点0和节点1，容量为100
    [1, 2, 200],  # 道路1连接节点1和节点2，容量为200
    [2, 3, 150],  # 道路2连接节点2和节点3，容量为150
    [3, 4, 100],  # 道路3连接节点3和节点4，容量为100
    [4, 0, 50],  # 道路4连接节点4和节点0，容量为50
    [0, 2, 80],  # 道路5连接节点0和节点2，容量为80
    [1, 3, 60],  # 道路6连接节点1和节点3，容量为60
    [2, 4, 40]  # 道路7连接节点2和节点4，容量为40
])

# 定义初始车流量分配
initial_flow = np.zeros(num_roads)

# 定义学习率和迭代次数
learning_rate = 0.1
num_iterations = 100

# 定义梯度下降函数
def gradient_descent(roads, flow, learning_rate, num_iterations):
    for i in range(num_iterations):
        # 计算每条道路的拥堵指数
        congestion_index = (flow - roads[:, 2]) ** 2

        # 计算损失函数的梯度
        gradient = 2 * (flow - roads[:, 2])

        # 更新车流量分配
        flow -= learning_rate * gradient

        # 打印每次迭代的损失函数值
        loss = np.sum(congestion_index)
        print(f"第 {i + 1} 次迭代：损失值 = {loss}")

    return flow

# 执行梯度下降优化算法
optimized_flow = gradient_descent(roads, initial_flow, learning_rate, num_iterations)

# 打印优化后的车流量分配结果
print("优化后的交通流量:")
for i in range(num_roads):
    print(f"道路 {i}: {optimized_flow[i]}")
