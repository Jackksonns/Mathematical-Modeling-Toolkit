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

def visualize_shortest_path(graph, source, target, shortest_path):
    plt.subplot(1, 2, 2)
    plt.title("Shortest Path")
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightgray', node_size=500)
    nx.draw_networkx_nodes(graph, pos, nodelist=[source, target], node_color='red', node_size=500)
    nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.7, edge_color='lightgray')
    nx.draw_networkx_edges(graph, pos, edgelist=shortest_path, width=2.0, alpha=0.7, edge_color='red')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

# 创建一个带有权值的有向图
G = nx.DiGraph()
G.add_edge(1, 2, weight=2)
G.add_edge(1, 3, weight=5)
G.add_edge(2, 3, weight=1)
G.add_edge(2, 4, weight=4)
G.add_edge(3, 4, weight=2)
G.add_edge(3, 5, weight=6)
G.add_edge(4, 5, weight=3)

# 指定源节点和目标节点
source = 1
target = 5

# 运行单源最短路径算法
shortest_path = nx.shortest_path(G, source, target, weight='weight')

# 绘制并列图形
plt.figure(figsize=(12, 6))

# 绘制原始图
visualize_graph(G, "Original Graph")

# 绘制最短路径
visualize_shortest_path(G, source, target, [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)])

# 调整图形布局
plt.tight_layout()

# 显示图形
plt.show()