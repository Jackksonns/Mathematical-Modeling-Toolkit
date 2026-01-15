"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# 生成数据
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 简单的线性分类

# 可视化数据
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', alpha=0.5)
plt.title('Original Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 训练模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 可视化决策边界
plt.figure(figsize=(10, 8))
plot_decision_regions(X=X_train, y=y_train, clf=gnb, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Naive Bayes Decision Boundary')
plt.show()

# 打印预测准确率
print('Accuracy: ', gnb.score(X_test, y_test))