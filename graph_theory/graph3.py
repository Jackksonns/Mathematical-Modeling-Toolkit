"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 创建一个简单的图
G = nx.Graph()
G.add_edge('A', 'B', weight=2)
G.add_edge('A', 'C', weight=3)
G.add_edge('B', 'D', weight=1)
G.add_edge('C', 'D', weight=5)
G.add_edge('C', 'E', weight=4)
G.add_edge('E', 'D', weight=1)

# 初始化距离字典，将每个节点的距离设置为无穷大
distances = {node: float('infinity') for node in G.nodes()}
distances['A'] = 0

# 使用 Dijkstra 算法找到最短路径
shortest_path = nx.dijkstra_path(G, 'A', 'D', weight='weight')

# 动态可视化建立最短路径的过程
fig, ax = plt.subplots()
pos = nx.spring_layout(G)
labels = {node: node for node in G.nodes()}
colors = ['blue' if d == 0 else 'red' for d in distances.values()]
nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=500, node_color=colors)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels, font_size=16, font_family='sans-serif')
plt.axis('off')
plt.show(block=False)

# 逐步显示每一步的建立过程
for i in range(len(shortest_path) - 1):
    next_node = shortest_path[i + 1]
    next_distance = distances[next_node]
    ax.text(pos[next_node][0], pos[next_node][1], f'Node {next_node}, Distance {next_distance}')
    distances[next_node] = next_distance + distances[shortest_path[i]] - G[shortest_path[i]][next_node]['weight']
    nx.draw_networkx_nodes(G, pos, nodelist=[shortest_path[i], next_node], node_size=500, node_color=['red', 'blue'])
    nx.draw_networkx_edges(G, pos)
    plt.pause(2)  # 增加延迟时间以减慢显示速度
    plt.clf()
plt.show(block=True)