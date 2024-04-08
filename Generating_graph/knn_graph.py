import dgl
import numpy as np
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from dgl.data.utils import save_graphs, load_graphs
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random

def Feature():
    a = '10'
    attacker = ['10.0.0.29', '10.0.0.30', '10.0.0.31', '10.0.0.32']
    file = open('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/testnew1.txt', 'r')
    lines = file.readlines()
    L = []
    for line in lines:
        if line != '-----\n' and line != 'Theendoftheround----------------\n' and line != '\n':
            L.append(eval(line))
    arr = np.zeros(shape=(len(L), 11, 1))
    # arr = np.zeros(shape=(len(L), 5))
    label = np.zeros(shape=(len(L), 1))
    for i in range(len(L)):
        arr[i, 0, 0] = L[i]['duration']
        arr[i, 1, 0] = L[i]['packet_count']
        arr[i, 2, 0] = L[i]['packet_grow']
        arr[i, 3, 0] = L[i]['byte_count']
        arr[i, 4, 0] = L[i]['byte_grow']
        arr[i, 5, 0] = L[i]['v_byte_flow']
        arr[i, 6, 0] = L[i]['asym_byte_rate']
        arr[i, 7, 0] = L[i]['asym_byte_rate_inter']
        arr[i, 8, 0] = L[i]['s_ip_ratio']
        arr[i, 9, 0] = L[i]['d_ip_ratio']
        arr[i, 10, 0] = L[i]['ip_tuple']
        if (L[i]['ipv4_src'] in attacker) or (L[i]['ipv4_dst'] in attacker):
            label[i] = 1

    random.seed(30)
    random.shuffle(arr)
    random.seed(30)
    random.shuffle(label)

    arr1 = min_max(arr)
    return arr1, newlabel


def min_max(x):
    x_max = np.max(x, axis=0)
    # x_mean = np.mean(x, axis=0)
    x_min = np.min(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x_max[j, 0] - x_min[j, 0] != 0:
            # if x_max[j] - x_min[j] != 0:
                # x[i,j] = float(x[i,j] - x_mean[j]) / float(x_max[j] - x_min[j])
                # x[i, j] = float(x[i, j] - x_min[j]) / float(x_max[j] - x_min[j])
                x[i, j, 0] = float(x[i, j, 0] - x_min[j, 0]) / float(x_max[j, 0] - x_min[j, 0])
    return x

def KNN_graph(data):
    # data = np.reshape(data, (data.shape[0], -1))
    graph = []
    distance = np.zeros(shape=(11, 11))
    for i in range(len(data)):
        for j in range(len(distance)):
            for k in range(len(distance)):
                distance[j, k] += (data[i, j, 0] - data[i, k, 0]) ** 2
        # A = kneighbors_graph(data[i], 3, mode='connectivity', metric='euclidean', include_self=True)
        # index = np.argwhere(A==1)
        # src = []
        # dst = []
        # for j in range(len(index)):
        #     src.append(index[j, 0])
        #     dst.append(index[j, 1])
        # g = dgl.graph((src, dst))
        # Feat = th.from_numpy(data[i])
        # Feat = Feat.to(th.float32)
        # g.ndata['feature'] = Feat
        # graph.append(g)
    distance = np.sqrt(distance)
    K = 3 # Knn --> K = 3
    src = []
    dst = []
    for i in range(len(distance)):
        for k in range(len(distance) - K):
            index = np.argwhere(distance[i] == np.max(distance[i]))
            distance[i, index] = -1
        for j in range(len(distance[i])):
            if distance[i, j] != -1:
                src.append(i)
                dst.append(j)
    for i in range(len(data)):
        g = dgl.graph((src, dst))
        Feat = th.from_numpy(data[i])
        Feat = Feat.to(th.float32)
        g.ndata['feature'] = Feat
        g = dgl.add_self_loop(g)
        graph.append(g)
    return graph


def ShowGraph(graph, nodeLabel, EdgeLabel):
    plt.figure(figsize=(11, 11))
    G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())  # 转换 dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)  # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = {index: "N:" + str(data) for index, data in
                   enumerate(node_data)}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}

    for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  # 将desc属性，显示在节点上
    edge_labels = nx.get_edge_attributes(G, EdgeLabel)  # 获取边的weights属性，

    edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in
                   edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上

    # print(G.edges.data())
    plt.show()


data, label = Feature()
graph = KNN_graph(data)
graph_labels = {'label': th.tensor(label)}
save_graphs("/home/ran/Desktop/openrainbow/openrainbow/GCN/data/knn(safive).bin", graph, graph_labels)
