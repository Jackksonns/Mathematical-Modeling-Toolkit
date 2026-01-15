"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def level_check(x, r=(0.1, 2.5)):
    # 级比检验
    n = len(x)
    lambda_x = [x[i-1] / x[i] for i in range(1, n)]
    return all(r[0] <= lambda_x[i] <= r[1] for i in range(n-1))

def GM_11(x0):
    # 建立GM(1,1)模型
    n = len(x0)
    x1 = x0.cumsum() # 一次累加
    z1 = (x1[:n-1] + x1[1:]) / 2.0 # 紧邻均值生成序列
    B = np.array([-z1, np.ones(n-1)]).T
    Yn = x0[1:].reshape((n-1, 1))
    [[a], [b]] = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Yn) # 计算参数
    f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) # 还原值
    return f, a, b


def check_predict(x0, f):
    n = len(x0)
    predict = [f(i+1) for i in range(n)]  # 将预测序列前移一位
    #predict[0] = x0[0]  # 使用原始序列的第一个值替换预测序列的第一个值
    e = x0 - np.array(predict)  # 残差
    relative_e = e / x0  # 相对误差
    lambda_k = [None] + [x0[i] / x0[i-1] for i in range(1, n)]
    lambda_k_hat = [None, None] + [(predict[i]-predict[i-1]) / (predict[i-1]-predict[i-2]) for i in range(2, n)]
    delta_k = [None if lk is None or lhk is None else abs(lk - lhk) for lk, lhk in zip(lambda_k, lambda_k_hat)]  # 级比偏差
    return predict, relative_e, delta_k


def create_df(x0, predict, e, delta_k, relative_e):
    # 创建一个包含原始值、模型值、残差、级比偏差和相对误差的pandas DataFrame
    df = pd.DataFrame({
        '原始值': x0,
        '模型值': predict,
        '残差': e,
        '级比偏差': delta_k,
        '相对误差': relative_e
    })
    return df

def plot_data(x0, predict):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(x0)), x0, 'o-', label='原始值')
    plt.plot(range(len(predict)), predict, 'r--', label='预测值')
    plt.xlabel('时间')
    plt.ylabel('值')
    plt.title('灰色预测模型')
    plt.legend()
    plt.show()

# Test with some data
x0 = np.array([71.1,72.4,72.4,72.1,71.4,72.0,71.6])
if level_check(x0):
    f, a, b = GM_11(x0)
    predict, relative_e, delta_k = check_predict(x0, f)
    e = x0 - predict
    df = create_df(x0, predict, e, delta_k, relative_e)
    plot_data(x0, predict)
else:
    print("级比检验不通过，不能使用灰色预测")
df
