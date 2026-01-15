"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建K近邻模型并进行训练
k = 3  # 设置K的值为3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 预测测试集的结果
y_pred = knn.predict(X_test)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')  # 绘制原始数据点
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x')  # 绘制预测的测试数据点
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Nearest Neighbors Classification')
plt.show()

# 输出预测结果和实际结果的比较
print("Predicted labels:", y_pred)
print("Actual labels:", y_test)