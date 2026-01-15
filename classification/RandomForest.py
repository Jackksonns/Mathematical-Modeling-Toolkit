"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 生成随机数据
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pre = model.predict(X_test)

# 将原始数据和预测数据可视化
plt.figure(figsize=(10, 5))

# 绘制原始数据的散点图
plt.subplot(121)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 绘制预测数据的散点图
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pre, cmap='viridis')
plt.title('Predicted Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 显示图形
plt.show()
