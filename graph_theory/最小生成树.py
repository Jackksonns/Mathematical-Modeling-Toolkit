"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, title):
    plt.subplot(1, 2, 1)
    plt.title(title)
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightgray', node_size=500)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

def visualize_minimum_spanning_tree(graph, minimum_spanning_tree):
    plt.subplot(1, 2, 2)
    plt.title("Minimum Spanning Tree")
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(minimum_spanning_tree, pos, width=2.0, alpha=0.7, edge_color='red')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

# 创建一个带有权值的无向图
G = nx.Graph()
G.add_edges_from([(1, 2, {'weight': 3}), (1, 3, {'weight': 2}), (2, 3, {'weight': 4}), (2, 4, {'weight': 1}),
                  (3, 4, {'weight': 5}), (3, 5, {'weight': 6}), (4, 5, {'weight': 7})])

# 运行最小生成树算法
minimum_spanning_tree = nx.minimum_spanning_tree(G)

# 绘制并列图形
plt.figure(figsize=(12, 6))

# 绘制原始图
visualize_graph(G, "Original Graph")

# 绘制最小生成树
visualize_minimum_spanning_tree(G, minimum_spanning_tree)

# 调整图形布局
plt.tight_layout()

# 显示图形
plt.show()