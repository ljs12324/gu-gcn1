import math
import random
from time import sleep

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import nn, tensor
# from d2l import torch as d2l
import os
import dgl
import numpy as np
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from dgl.data.utils import save_graphs, load_graphs
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def Feature():
    # dataprocess=('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/flow_feature.txt')
    attacker = ['10.0.0.29', '10.0.0.30', '10.0.0.31', '10.0.0.32']
    file = open('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/testnew.txt', 'r')
    lines = file.readlines()
    L = []
    for line in lines:
        if line != '-----\n' and line != 'Theendoftheround----------------\n' and line != '\n':
            L.append(eval(line))
    arr = np.zeros(shape=(len(L), 11, 12))
    # arr = np.zeros(shape=(len(L), 5))
    label = np.zeros(shape=(len(L), 1))
    for i in range(len(L)):
        arr[i, 0] = L[i]['duration']
        arr[i, 1] = L[i]['packet_count']
        arr[i, 2] = L[i]['packet_grow']
        arr[i, 3] = L[i]['byte_count']
        arr[i, 4] = L[i]['byte_grow']
        arr[i, 5] = L[i]['v_byte_flow']
        arr[i, 6] = L[i]['asym_byte_rate']
        arr[i, 7] = L[i]['asym_byte_rate_inter']
        arr[i, 8] = L[i]['s_ip_ratio']
        arr[i, 9] = L[i]['d_ip_ratio']
        arr[i, 10] = L[i]['ip_tuple']
        if (L[i]['ipv4_src'][0] in attacker) or (L[i]['ipv4_dst'][0] in attacker):
            label[i] = 1

    random.seed(30)
    random.shuffle(arr)
    random.seed(30)
    random.shuffle(label)

    newarr = arr[10000:39515]
    newlabel = label[10000:39515]
    arr1, arr2 = min_max1(newarr)
    return arr1, arr2, newlabel

def min_max(x):
    # x_max = []
    # x_min = []
    # for i in range(11):
    #     x_max.append([])
    #     x_min.append([])
    #     for j in range(12):
    #         x_max[i].append(np.max(x[:,i,j]))
    #         x_min[i].append(np.min(x[:,i,j]))
    x_max = np.max(x, axis=0)
    x_max = np.max(x_max,axis=1)
    x_min = np.min(x, axis=0)
    x_min = np.min(x_min, axis=1)
    # x_max = np.array(x_max)
    # x_min = np.array(x_min)
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                if x_max[j] - x_min[j] != 0:
                    x[i, j,k] = float(x[i, j,k] - x_min[j]) / float(x_max[j] - x_min[j])
    return x

def min_max1(x):
    # x_max = []
    # x_min = []
    # for i in range(11):
    #     x_max.append([])
    #     x_min.append([])
    #     for j in range(12):
    #         x_max[i].append(np.max(x[:,i,j]))
    #         x_min[i].append(np.min(x[:,i,j]))
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_arr = []
    x_all = []
    # x_max = np.array(x_max)
    # x_min = np.array(x_min)
    for i in range(len(x)):
        x_arr.append([])
        for j in range(len(x[i])):
            if x_max[j,0] - x_min[j,0] !=0:
                ress = float(x[i, j,0] - x_min[j,0]) / float(x_max[j,0] - x_min[j,0])
                x_arr[i].append([ress])
    for i in range(len(x)):
        x_all.append([])
        for j in range(len(x[i])):
            x_all[i].append([])
            for k in range(len(x[i][j])):
                x_all[i][j].append(float(x[i, j, k] - x_min[j,k]) / float(x_max[j,k] - x_min[j,k]))
            # for k in range(len(x[i][j])):
            #     if x_max[j] - x_min[j] != 0:
            #         x[i, j,k] = float(x[i, j,k] - x_min[j]) / float(x_max[j] - x_min[j])
    x_arr = np.array(x_arr)
    x_all = np.array(x_all)
    return x_arr,x_all


def gray_relation_coefficient(x, y, delta=0.5):
    """
    计算灰色关联系数
    :param x: 序列x
    :param y: 序列y
    :param delta: 分辨系数，默认为0.5
    :return: 灰色关联系数
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("输入序列的长度不一致")

    # 计算序列的累加值
    cumsum_x = np.cumsum(x)
    cumsum_y = np.cumsum(y)

    # 计算离散累加生成序列
    v_x = np.array(
        [delta * cumsum_x[i] + (1 - delta) * cumsum_x[i - 1] if i > 0 else delta * cumsum_x[i] for i in range(n)])
    v_y = np.array(
        [delta * cumsum_y[i] + (1 - delta) * cumsum_y[i - 1] if i > 0 else delta * cumsum_y[i] for i in range(n)])
    # 计算关联系数
    rho = 1 - (np.abs(v_x - v_y) / (np.max(np.abs(v_x - v_y)) + 0.000000000000000001))

    # 返回平均关联系数作为最终结果
    return np.mean(rho)

def KNN_graph(data):
    # data = np.reshape(data, (data.shape[0], -1))
    graph = []
    distance = np.zeros(shape=(11, 11))
    # edge_value = np.zeros(shape=(11, 11, 3))
    # for i in range(len(data)):
    #     for j in range(len(distance)):
    #         for k in range(len(distance)):
    #             print(data[:, 0, 0])
    #
    #             print(data[:, 1, 0])
    #             # vec1 = th.Tensor(data[:, 0, 0])
    #             # vec2 = th.Tensor(data[:, 1, 0])
    #             vec1 = data[:, j, 0].reshape(1, -1)
    #             vec2 = data[:, k, 0].reshape(1, -1)
    #             print(len(data[:, 0, 0]))
    #             #cos_sim = F.cosine_similarity(vec1.T, vec2.T, dim=0)
    #             cos_sim = cosine_similarity(vec1, vec2)
    #             print(cos_sim)
    #             distance[j, k] += cos_sim
    src1 = []
    dst1 = []
    gray_list = []
    for i in range(len(data)):
        src1.append([])
        dst1.append([])
        gray_list.append([])
        for j in range(len(data[i])):
            for k in range(len(data[i])):
                result = gray_relation_coefficient(data[i][j],data[i][k])
                if result < 0.5:
                    pass
                else:
                    src1[i].append(j)
                    dst1[i].append(k)
                    gray_list[i].append(result)

    for i in range(len(data)):
        g = dgl.graph((src1[i], dst1[i]))
        Feat = th.from_numpy(data[i])
        Feat = Feat.to(th.float32)
        g.ndata['feature'] = Feat
        dis = tensor(gray_list[i])
        g.edata['value'] = dis
        g = dgl.add_self_loop(g)
        graph.append(g)
    return graph

    # for j in range(len(distance)):
    #     for k in range(len(distance)):
    #         # print(data[:, 0, 0])
    #         # print(data[:, 1, 0])
    #         # vec1 = th.Tensor(data[:, 0, 0])
    #         # vec2 = th.Tensor(data[:, 1, 0])
    #         vec1 = data[:, j, 0].reshape(1, -1)
    #         vec2 = data[:, k, 0].reshape(1, -1)
    #         # print(len(data[:, 0, 0]))
    #         # cos_sim = F.cosine_similarity(vec1.T, vec2.T, dim=0)
    #         # cos_sim = F.cosine_similarity(tensor(vec1), tensor(cos_vec2), dim=0)
    #         cos_sim =cosine_similarity(vec1, vec2)
    #         distance[j, k] += cos_sim
    # distance = np.sqrt(distance)
    # K = 3 # Knn --> K = 3
    # src = []
    # dst = []
    #
    #
    #
    # # print(len(distance))
    # for i in range(len(distance)):
    #     for k in range(len(distance) - K):
    #         index = np.argwhere(distance[i] == np.min(distance[i]))
    #         distance[i, index] = 10000
    #     for j in range(len(distance[i])):
    #         if distance[i, j] != 10000:
    #             src.append(i)
    #             dst.append(j)
    #             # if str(i) in attention_value.keys():
    #             #     value1 = attention_value[str(i)]
    #             # value1=attention_value.get(str(i))
    #             # if str(j) in attention_value:
    #             #     value2 = attention_value[str(j)]
    #             # value2 = attention_value.get(str(j))
    #             # edge_data = [value1, distance[i,j],value2]
    #             #edge_data = np.mat(edge_data)
    #             # print("Original dataset = {}*{}".format(edge_data.shape[0], edge_data.shape[1]))
    #             # pca = PCAcomponent(edge_data, 1)
    #             # pca.fit()
    #             # print(pca.low_dataMat)
    #             # print(pca.variance_ratio)
    #             # dis.append(distance[i,j])
    #             # dis.append(edge_data)
    #             # edge_value[i, j] += edge_data
    #             # for i in range(len(edge_value)):
    #             #     for j in range(len(edge_value)):
    # distance_value = [[] for i in range(len(src))]
    # value1=[]
    # value2=[]
    # for i in range(len(src)):
    #     s=src[i]
    #     d=dst[i]
    #     # value1.append(attention_value.get(str(s)))
    #     # value2.append(attention_value.get(str(d)))
    #     for j in range(len(data)):
    #         cos_value1=data[j, s, 0].reshape(1, -1)   #data[j][s]
    #         cos_value2=data[j, d, 0].reshape(1, -1)   #data[j][d]
    #         dis_value =F.cosine_similarity(tensor(cos_value1), tensor(cos_value2), dim=0)
    #         dis_value = dis_value.item()
    #         distance_value[i].append(dis_value)
    #
    #
    # for i in range(len(data)):
    #     dis = []
    #     g = dgl.graph((src, dst))
    #     Feat = th.from_numpy(data[i])
    #     Feat = Feat.to(th.float32)
    #     g.ndata['feature'] = Feat
    #     # print(type(Feat))
    #     # print(g.ndata['feature'])
    #     for j in range(len(src)):
    #         # edge_data = [value1[j], distance_value[j][i], value2[j]]
    #         edge_data = [distance_value[j][i]]
    #         dis.append(edge_data)
    #     # print(type(dis))
    #     dis = tensor(dis)
    #     g.edata['value'] = dis
    #     # print(g.edata['value'])
    #         # for s in range(len(distance)):
    #         #     for d in range(len(distance[i])):
    #         #         edge_value=tensor(edge_value)
    #         #         print(edge_value[s, d])
    #         #         print(type(edge_value))
    #         #         if edge_value[s,d]!=[0.,0.,0.]:
    #         #             g.edata['value']=edge_value[s,d]
    #         #             print(g.edata['value'])
    #     g = dgl.add_self_loop(g)
    #     graph.append(g)
    #     # nx.draw(g.to_networkx(), with_labels=True)
    #     # # sleep(5000)
    #     # plt.show()
    # return graph

def save(file_path, graph, graph_labels):
    # 将图形数据和标签转换为NumPy数组
    graph_array = np.array(graph)
    labels_array = np.array(graph_labels)

    # 使用NumPy保存为二进制文件
    with open(file_path, 'wb') as file:
        np.save(file, graph_array)
        np.save(file, labels_array)

if __name__ == '__main__':
    data,data1, label = Feature()
    data_labels = {'label': th.tensor(label)}
    np.save('dataset_with_labels.npy', {'data': data1, 'labels': data_labels})
    graph = KNN_graph(data1)
    graph_labels = {'label': th.tensor(label)}
    save_graphs("/home/ran/Desktop/openrainbow/openrainbow/GCN/data/graph_data(safive).bin", graph, graph_labels)