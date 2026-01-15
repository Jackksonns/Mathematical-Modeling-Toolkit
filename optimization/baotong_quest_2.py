"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 模拟参数
mean_demand = 30  # 需求均值
std_demand = 10  # 需求标准差
unit_cost = 1  # 单位成本
shortage_cost = 2  # 短缺成本
order_cost = 10  # 订货成本

# 模拟需求
demand_samples = np.random.normal(mean_demand, std_demand, size=100)

# 计算总成本的函数
def total_cost(order_quantity):
    if order_quantity <= 0:  # 基准情况，停止递归
        return 0, 0, 0, 0

    # 计算库存成本
    inventory_cost = 0.5 * unit_cost * order_quantity

    # 计算短缺成本
    shortage_prob = norm.cdf(order_quantity, mean_demand, std_demand)
    shortage_cost_total = shortage_cost * np.sum(np.maximum(0, demand_samples - order_quantity))

    # 计算订货成本
    order_cost_total = order_cost * np.ceil(np.mean(demand_samples) / order_quantity)

    # 返回总成本以及各个子成本
    return inventory_cost, shortage_cost_total, order_cost_total, inventory_cost + shortage_cost_total + order_cost_total

# 在一定范围内尝试不同的订货量
order_quantities = np.arange(1, 100, 1)
costs = [total_cost(q) for q in order_quantities]

# 找到最小总成本对应的订货量索引
optimal_order_quantity_index = np.argmin([cost[3] for cost in costs])
optimal_order_quantity = order_quantities[optimal_order_quantity_index]

# 输出各个子成本的数值
inventory_cost = costs[optimal_order_quantity_index][0]
shortage_cost = costs[optimal_order_quantity_index][1]
order_cost = costs[optimal_order_quantity_index][2]
total_cost = costs[optimal_order_quantity_index][3]

print(f"Optimal Order Quantity: {optimal_order_quantity}")
print(f"Inventory Cost: {inventory_cost}")
print(f"Shortage Cost: {shortage_cost}")
print(f"Order Cost: {order_cost}")
print(f"Total Cost: {total_cost}")

# 绘制成本与订货量的关系
plt.plot(order_quantities, [cost[3] for cost in costs], label='Total Cost')
plt.axvline(x=optimal_order_quantity, color='r', linestyle='--', label='Optimal Order Quantity')
plt.xlabel('Order Quantity')
plt.ylabel('Total Cost')
plt.legend()
plt.title('Total Cost vs. Order Quantity')
plt.show()
