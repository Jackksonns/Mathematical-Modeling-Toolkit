"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

# 生成随机的交叉口信号灯时间数据
np.random.seed(45)  # 设置随机数种子，确保每次运行结果一致
num_intersections = 5  # 交叉口数量
signal_times = np.random.randint(25, 60, size=num_intersections)  # 生成随机信号灯时间数据，时长在10-30秒之间


# 定义交叉口的车辆等待时间函数
def waiting_time(signal_times):
    # 这里简化为一个简单的函数，实际问题可能更为复杂
    return np.sum(signal_times) + np.random.normal(scale=5)  # 计算所有交叉口信号灯总时长，并加上一个正态分布的随机扰动


# 定义梯度计算函数
def gradient(signal_times):
    epsilon = 0.01  # 梯度计算中的小量
    grad = np.zeros_like(signal_times)  # 初始化梯度数组

    for i in range(len(signal_times)):  # 遍历每个交叉口的信号灯时间
        signal_times_copy = signal_times.copy()  # 复制当前的信号灯时间数组
        signal_times_copy[i] += epsilon  # 对第i个交叉口的信号灯时间增加一个小的量
        grad[i] = (waiting_time(signal_times_copy) - waiting_time(signal_times)) / epsilon  # 计算梯度

    return grad  # 返回梯度数组


# 将signal_times转换为浮点数类型
signal_times = signal_times.astype(float)  # 将信号灯时间数组转换为浮点数类型

# 使用梯度下降进行优化
learning_rate = 0.1  # 学习率
num_iterations = 1000  # 迭代次数

for i in range(num_iterations):  # 进行迭代优化
    signal_times -= learning_rate * gradient(signal_times)  # 使用梯度下降法更新信号灯时间数组

# 可视化输出结果
plt.plot(signal_times, label='Optimized Signal Times')  # 绘制优化后的信号灯时间数组
plt.xlabel('Intersection')  # x轴标签：交叉路口
plt.ylabel('Signal Time')  # y轴标签：信号灯时长
plt.title('Optimized Traffic Signal Times')  # 图表标题：优化后的信号灯时间长度
plt.legend()  # 显示图例
plt.show()  # 显示图表
