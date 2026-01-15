"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可重现
np.random.seed(42)

# 假设历史股票收益率数据
stock_returns = np.random.randn(252, 4) * 0.01  # 252个交易日，4支股票，每日收益率假设为正态分布

# 假设股票的名称
stock_names = ['股票A', '股票B', '股票C', '股票D']

# 生成DataFrame方便计算
df_returns = pd.DataFrame(stock_returns, columns=stock_names)

# 计算年化收益率和协方差矩阵
annual_return = df_returns.mean() * 252
cov_matrix = df_returns.cov() * 252

# 初始化空列表存储回报、波动率和投资组合权重
portfolio_returns = []
portfolio_volatilities = []
sharpe_ratios = []
weight_list = []

# 模拟的投资组合数量
num_portfolios = 10000

for _ in range(num_portfolios):
    weights = np.random.random(len(stock_names))  # 生成随机权重
    weights /= np.sum(weights)  # 权重归一化

    # 计算投资组合年化收益率和波动率
    returns = np.dot(weights, annual_return)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 计算夏普比率
    sharpe_ratio = returns / volatility

    portfolio_returns.append(returns)
    portfolio_volatilities.append(volatility)
    sharpe_ratios.append(sharpe_ratio)
    weight_list.append(weights)

# 创建DataFrame
portfolios = pd.DataFrame({
    '返回': portfolio_returns,
    '波动率': portfolio_volatilities,
    '夏普比率': sharpe_ratios
})

# 将权重数据加入DataFrame
for i, stock in enumerate(stock_names):
    portfolios[stock + ' 权重'] = [weight[i] for weight in weight_list]

# 找到夏普比率最高的投资组合
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe_portfolio = portfolios.iloc[max_sharpe_idx]

# 可视化所有投资组合和最优投资组合
plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='viridis')
plt.colorbar(label='夏普比率')
plt.xlabel('波动率')
plt.ylabel('预期收益')
plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], marker='*', color='r', s=500, label='最优组合')
plt.legend()
plt.show()

print("最优组合的权重和指标：")
print(max_sharpe_portfolio)