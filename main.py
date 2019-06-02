import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# #作业一
# #有向图
# G = nx.DiGraph()
# n = 0
# with open ('www.dat.gz.txt') as f:
#     for line in f:
#         n += 1
#         x, y = line.rstrip().split(' ')
#         G.add_edge(x,y)
# print(nx.info(G))
# print(nx.density(G))


#作业二
from collections import defaultdict
import numpy as np
def plotInOutDegreeDistribution(G):
    in_degs = defaultdict(int)
    out_degs = defaultdict(int)
    for i in dict(G.in_degree()).values(): in_degs[i]+=1
    for i in dict(G.out_degree()).values(): out_degs[i]+=1
    
    fig = plt.figure(figsize=(25, 5),facecolor='white')
    a_fig = plt.subplot(1,2,1)
    items = sorted ( in_degs.items () )
    x, y = np.array(items).T
    y_sum = np.sum(y)
    y = [float(i)/y_sum for i in y]
    plt.plot(x, y, 'b-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['InDegree'])
    plt.xlabel('$K$', fontsize = 20)
    plt.ylabel('$P(K)$', fontsize = 20)
    plt.title('$InDegree\,Distribution$', fontsize = 20)

    b_fig = plt.subplot(1,2,2)
    items = sorted ( out_degs.items () )
    x, y = np.array(items).T
    y_sum = np.sum(y)
    y = [float(i)/y_sum for i in y]
    plt.plot(x, y, 'b-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['OutDegree'])
    plt.xlabel('$K$', fontsize = 20)
    plt.ylabel('$P(K)$', fontsize = 20)
    plt.title('$OutDegree\,Distribution$', fontsize = 20)
    plt.show()

#print(nx.in_degree_centrality(G))
#plotInOutDegreeDistribution(G)

# import time
# Ns = [i*10 for i in range(1,3)]
# ds = []
# start = time.time()
# for N in Ns:
#     #print(N)
#     BA= nx.random_graphs.barabasi_albert_graph(N,2)
#     d = nx.average_shortest_path_length(BA)
#     ds.append(d)
# end = time.time()
# print("spend time:",end - start,"s")
# plt.plot(Ns, ds, 'r-o')
# plt.xlabel('$N$', fontsize = 20)
# plt.ylabel('$<d>$', fontsize = 20)
# #plt.xscale('log')
# plt.show()

import networkx as nx
import matplotlib.pyplot as plt
RG = nx.random_graphs.random_regular_graph(4,200)
WS = nx.random_graphs.watts_strogatz_graph(200,4,0.3)
print(len(WS.edges()))  
print(len(RG.edges()))
#生成包含200个节点、每个节点4个近邻、随机化重连概率为0.3的小世界网络
# 首先每个节点会有4个近邻边，然后以0.3的概率去确定是否重连这条边，即保持边的一个节点不变，另一个节点随机选择
pos = nx.spring_layout(WS)          
#定义一个布局，此处采用了circular布局方式
nx.draw(WS,pos,with_labels=False,node_size = 30)  
#绘制图形
plt.show()