"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import networkx as nx
import matplotlib.pyplot as plt

def visualize_shortest_path(graph, source, shortest_paths):
    # 创建一个空的图形对象
    plt.figure(figsize=(8, 6))

    # 绘制原始图形
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightgray', node_size=500)

    # 标记源节点
    nx.draw_networkx_nodes(graph, pos, nodelist=[source], node_color='red', node_size=500)

    # 绘制最短路径
    for target, path in shortest_paths.items():
        if target != source:
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, width=2.0, alpha=0.7)

    # 显示图形
    plt.axis('off')
    plt.show()

# 创建一个有向图
G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6)])

# 运行最短路径算法
source_node = 1
shortest_paths = nx.single_source_shortest_path(G, source_node)

# 可视化展示最短路径
visualize_shortest_path(G, source_node, shortest_paths)