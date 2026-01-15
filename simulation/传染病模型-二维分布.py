"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

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

# 创建动画
fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('SARS Spread in a Population')

# 更新函数
def update(frame):
    infected_coords = np.random.rand(int(I[frame]*1000), 2)
    susceptible_coords = np.random.rand(int(S[frame]*1000), 2)
    recovered_coords = np.random.rand(int(R[frame]*1000), 2)

    scatter.set_offsets(np.vstack([infected_coords, susceptible_coords, recovered_coords]))
    scatter.set_array(np.concatenate([np.full(len(infected_coords), 1), np.full(len(susceptible_coords), 0), np.full(len(recovered_coords), 2)]))

# 创建动画
animation = FuncAnimation(fig, update, frames=len(t), interval=50)

plt.show()
