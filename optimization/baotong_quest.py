"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 模拟参数
mean_demand = 30
std_demand = 10
unit_cost = 1
shortage_cost = 2
order_cost = 10

# 模拟需求
demand_samples = np.random.normal(mean_demand, std_demand, size=100)


# 计算总成本的函数
def total_cost(order_quantity):
    # 计算库存成本
    inventory_cost = 0.5 * unit_cost * order_quantity

    # 计算短缺成本
    shortage_prob = 1.0 - norm.cdf(order_quantity, mean_demand, std_demand)
    shortage_cost_total = shortage_cost * np.sum(np.maximum(0, demand_samples - order_quantity))

    # 计算订货成本
    order_cost_total = order_cost * np.ceil(np.mean(demand_samples) / order_quantity)

    # 返回总成本
    return inventory_cost + shortage_cost_total + order_cost_total



# 在一定范围内尝试不同的订货量
order_quantities = np.arange(1, 100, 1)
costs = [total_cost(q) for q in order_quantities]

# 找到最小总成本对应的订货量
optimal_order_quantity = order_quantities[np.argmin(costs)]



# 绘制成本与订货量的关系
plt.plot(order_quantities, costs, label='Total Cost')
plt.axvline(x=optimal_order_quantity, color='r', linestyle='--', label='Optimal Order Quantity')
plt.xlabel('Order Quantity')
plt.ylabel('Total Cost')
plt.legend()
plt.title('Total Cost vs. Order Quantity')
plt.show()

print(f"Optimal Order Quantity: {optimal_order_quantity}")
