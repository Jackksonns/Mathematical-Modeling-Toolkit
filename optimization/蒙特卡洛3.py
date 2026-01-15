"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import random
import statistics
import matplotlib.pyplot as plt
import numpy as np


def estimate_average_commute_time(n_samples=1000):
    """
    使用蒙特卡洛方法估算城市的平均通勤时间。
    :param n_samples: 抽样调查的数量
    :return: 平均通勤时间的近似值
    """
    # 假设我们知道整个城市通勤时间的分布范围，例如从10分钟到90分钟
    min_commute_time = 10
    max_commute_time = 90

    # 进行抽样调查，随机选择n_samples个通勤者，并记录他们的通勤时间
    sample_commute_times = []
    for _ in range(n_samples):
        commute_time = random.uniform(min_commute_time, max_commute_time)
        sample_commute_times.append(commute_time)

        # 计算抽样样本的平均通勤时间，作为整个城市平均通勤时间的近似值
    average_commute_time = statistics.mean(sample_commute_times)
    return average_commute_time, sample_commute_times


# 使用蒙特卡洛方法估算平均通勤时间，并获取抽样样本数据
average_commute_time, sample_commute_times = estimate_average_commute_time()

# 创建直方图展示平均通勤时间的分布
bins = np.arange(0, 101, 10)  # 设置直方图的bin数和范围
counts, _ = np.histogram(sample_commute_times, bins=bins)  # 使用抽样样本数据生成直方图

plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label='Monte Carlo Estimate')  # 绘制直方图
plt.axvline(x=average_commute_time, color='r', linestyle='--', label='Estimated Average Commute Time')  # 绘制估算平均通勤时间线
plt.legend()  # 显示图例
plt.title('Average Commute Time Distribution')  # 设置图表标题
plt.xlabel('Commute Time (minutes)')  # 设置x轴标签
plt.ylabel('Frequency')  # 设置y轴标签
plt.show()  # 显示图表