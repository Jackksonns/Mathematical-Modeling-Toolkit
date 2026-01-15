"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt

def epidemic_model(t, S0, I0, R0, beta, gamma):
    """
    病毒传播模型函数

    参数:
        t: 时间
        S0: 初始易感者数量
        I0: 初始感染者数量
        R0: 初始康复者数量
        beta: 传染率
        gamma: 康复率

    返回值:
        随时间变化的易感者、感染者、康复者数量
    """
    N0 = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    for i in range(1, len(t)):
        S_new = S[i-1] - (beta * S[i-1] * I[i-1]) / N0
        I_new = I[i-1] + (beta * S[i-1] * I[i-1]) / N0 - gamma * I[i-1]
        R_new = R[i-1] + gamma * I[i-1]
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
    return S, I, R

# 参数设置
S0 = 900  # 初始易感者数量
I0 = 100  # 初始感染者数量
R0 = 0    # 初始康复者数量
beta = 0.3  # 传染率
gamma = 0.1  # 康复率

# 时间范围
t = np.linspace(0, 100, 100)

# 计算随时间变化的易感者、感染者、康复者数量
S, I, R = epidemic_model(t, S0, I0, R0, beta, gamma)

# 绘制人群数量变化曲线
plt.plot(t, S, label='s', linestyle='-', linewidth=2)
plt.plot(t, I, label='I', linestyle='--', linewidth=2)
plt.plot(t, R, label='R', linestyle=':', linewidth=2)
plt.xlabel('time', fontsize=12)
plt.ylabel('num', fontsize=12)
plt.title('model', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()