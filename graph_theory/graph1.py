"""
Mathematical-Modeling-Toolkit
Repository: https://github.com/Jackksonns/Mathematical-Modeling-Toolkit
"""

import networkx as nx
import matplotlib.pyplot as plt

# 创建一个图
G = nx.Graph()

# 添加节点
G.add_node("A")
G.add_node("B")
G.add_node("C")
G.add_node("D")
G.add_node("E")

# 添加边和权重
G.add_edge("A", "B", weight=1)
G.add_edge("A", "C", weight=2)
G.add_edge("B", "D", weight=3)
G.add_edge("C", "D", weight=1)
G.add_edge("C", "E", weight=4)
G.add_edge("D", "E", weight=2)

# 定义起点和终点
start = "A"
end = "E"

# 使用Dijkstra的算法找到最短路径
shortest_path = nx.dijkstra_path(G, start, end, weight='weight')

# 可视化输出
pos = nx.spring_layout(G)  # 使用spring_layout来得到节点位置
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='w', node_shape='o')  # 绘制节点
nx.draw_networkx_edges(G, pos, alpha=0.5)  # 绘制边
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')  # 添加节点标签
plt.axis('on')  # 不显示坐标轴
plt.show()  # 显示图

# 在图上标出最短路径和起点、终点
nx.draw_networkx_nodes(G, pos, nodelist=[start, end], node_color='r', node_size=1000)
nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], edge_color='r', width=3)
nx.draw_networkx_labels(G, pos, {'起点': start, '终点': end}, font_size=20, font_family='sans-serif')
plt.axis('on')  # 不显示坐标轴
plt.show()  # 显示图