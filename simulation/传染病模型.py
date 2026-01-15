"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义SIR模型
def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# 模型参数
beta = 0.3  # 传播速率

gamma = 0.1  # 恢复速率

# 初始条件
S0 = 0.99  # 初始易感人群比例
I0 = 0.01  # 初始感染人群比例
R0 = 0.0   # 初始康复人群比例

# 总时间
t = np.linspace(0, 200, 1000)

# 解ODE方程
solution = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
S, I, R = solution.T

# 绘制图表
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model for SARS Spread')
plt.legend()
plt.show()
