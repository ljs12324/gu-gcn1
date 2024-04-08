import math
import random
from time import sleep

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import nn, tensor
from d2l import torch as d2l
import os
import dgl
import numpy as np
import torch as th
import pickle as pkl
import math
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from dgl.data.utils import save_graphs, load_graphs
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:#不设置时，取全部值的 softmax
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape #将shape保存下来，以便取⽤其中的⾏列的维度数，以及最终恢复原样
        if valid_lens.dim() == 1:  #当valid_lens为⼀维
            #若x的维度为(2, 2, 4)得到第⼆个维度的数值2,并将valid_lens复制2次，得到⼀个
            valid_lens=valid_lens
            #valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            #经过这⼀步[2, 3]会变为[2, 2, 3, 3]
        else:
            valid_lens = valid_lens.reshape(-1)  #直接将其变为⼀维
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        # X.reshape(-1, shape[-1])将展开为n⾏4列,n在这⾥为2*2,形状为(4, 4)再对每⼀⾏进⾏2, 2, 3, 3的掩码操作
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) #得到的X是⼀个展开的⼆维张量
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def Normalization1(x):
    if len(x) == 1:
        # for i in range(len(x[0])):
            # x[0,i] = float(x[0,i] - )
        return x
    x_max = np.max(x, axis=0)
    # x_mean = np.mean(x, axis=0)
    x_min = np.min(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x_max[j] - x_min[j] != 0:
                # x[i,j] = float(x[i,j] - x_mean[j]) / float(x_max[j] - x_min[j])
                x[i, j] = float(x[i, j] - x_min[j]) / float(x_max[j] - x_min[j])
    return x

def Normalization2(self, x):
    x_max = np.max(x)
    x_min = np.min(x)
    #x_mean = np.mean(x)
    for i in range(len(x)):
        if x_max - x_min != 0:
            x[i] = float(x[i] - x_min) / float(x_max - x_min)
    return x

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout,**kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        #输⼊k维输出h维
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        #输⼊q维输出h维
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        #输⼊h维输出1维
        self.w_v = nn.Linear(num_hiddens,1, bias=False)
        #以p=dropout的概率进⾏正则化
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens):
      """
        :param valid_lens: 对每⼀个query 考虑前多少个key-value对
        :return:
      """
      #queries维度(bathc_size, q_num, h)  keys维度(bathc_size, k_num, h)
      queries, keys = self.W_q(queries), self.W_k(keys)
      #在维度扩展后，(在这⾥需要将每⼀个query和每⼀个key加在⼀起)
      # queries的形状：(batch_size，查询的个数，1，num_hidden)
      # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
      #使⽤⼴播⽅式进⾏求和
      features = queries.unsqueeze(2)+ keys.unsqueeze(1)
      #得到的features维度为(bathc_size, q_num, k_num, h)相当于每个q和k都做了求和
      features = torch.tanh(features) #激活
      # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
      # scores的形状：(batch_size，查询的个数，“键 -值”对的个数)
      scores = self.w_v(features).squeeze(-1)#squeeze(-1)把(batch_size, q, k, 1)最后有⼀个维度上的1去掉
      #print(scores)
      self.attention_weights = masked_softmax(scores, valid_lens)   #过滤掉不需要的k-v对
      # bmm为批量矩阵乘法,其中第⼀个参数的形状为：(batch_size, q, k)
      # values的形状：(batch_size, k, v)⼆者进⾏批量矩阵乘积得到(b, q, v)
      return torch.bmm(self.dropout(self.attention_weights), values)
    # 数据预处理
def dataprocess(data,line,row):
        #Q值得处理
        qu_data= torch.tensor(data, dtype=torch.float)
        #key值得处理
        keydata = data[:, 0]
        key_data = []
        for i in range(0,row):
            keydata = data[:, i]
            key_data.append(keydata.tolist())
        key_data=torch.tensor(key_data)
        queries=qu_data.view(line,1,row)
        keys=key_data.view(1,row,line)
        values= qu_data.view(line,row,1)
        valid_lens = torch.tensor([row, row, row, row, row, row, row, row, row,row,row])
        return queries,keys,values,valid_lens
        # attention = AdditiveAttention(key_size=line, query_size=row, num_hiddens=line,
        #                               dropout=0.1)

class PCAcomponent(object):
    def __init__(self, X, N=3):
        self.X = X
        self.N = N
        self.variance_ratio = []   #方差比率
        self.low_dataMat = []      #low_dataMat:低数据峰值

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)    #求平均值
        dataMat = self.X - X_mean
        # 另一种计算协方差矩阵的方法：dataMat.T * dataMat / dataMat.shape[0]
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)    #求协方差矩阵
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)     #将特征值从小到大排序
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        self.low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵
        # reconMat = (low_dataMat * small_eigVect.I) + X_mean  # 重构数据
        # 输出每个维度所占的方差百分比
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self

#
# d2l.show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),
#                   xlabel='Keys', ylabel='Queries')
# plt.show()
def attention(data1):
    # ATTENTION

    data1 = np.array(data1)
    data1 = Normalization1(data1)
    line = data1.shape[0]
    row = data1.shape[1]
    queries, keys, values, valid_lens = dataprocess(data1, line, row)
    attention = AdditiveAttention(key_size=line, query_size=row, num_hiddens=line,
                                  dropout=0.1)
    attention.eval()  # 开启评估模式
    a = attention(queries, keys, values, valid_lens)
    a = a.reshape(11)
    W_sum = sum(a)
    a = a / W_sum
    a = a.view(11, 1)
    return a

def Feature():
    # dataprocess=('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/flow_feature.txt')
    attacker = ['10.0.0.29', '10.0.0.30', '10.0.0.31', '10.0.0.32']
    # file = open('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/graph_feature1.txt', 'r')
    file = open('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data_more/testnew1.txt', 'r')
    lines = file.readlines()
    L = []
    for line in lines:
        if line != '-----\n' and line != 'The end of the round----------------\n' and line != '\n':
        # if line != '-----\n' and line != 'The end of the round----------------\n' and line != '\n':
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


    newarr = arr[10000:39515]
    newlabel = label[10000:39515]
    arr1 = min_max(newarr)
    # np.save('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/random_data/graph_feature1_label.npy', label)
    # np.save('/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/random_data/graph_feature1_arr.npy', arr)
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
    print(np.max(x[:,6]))
    return x

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
    for j in range(len(distance)):
        for k in range(len(distance)):
            # print(data[:, 0, 0])
            # print(data[:, 1, 0])
            # vec1 = th.Tensor(data[:, 0, 0])
            # vec2 = th.Tensor(data[:, 1, 0])
            vec1 = data[:, j, 0].reshape(1, -1)
            vec2 = data[:, k, 0].reshape(1, -1)
            # print(len(data[:, 0, 0]))
            # cos_sim = F.cosine_similarity(vec1.T, vec2.T, dim=0)
            # cos_sim = F.cosine_similarity(tensor(vec1), tensor(cos_vec2), dim=0)
            cos_sim =cosine_similarity(vec1, vec2)
            distance[j, k] += cos_sim
    distance = np.sqrt(distance)
    src = []
    dst = []
    # K=3
    # print(len(distance))
    for i in range(len(distance)):
        length=len(distance[i,:])
        print(distance[i,:])
        average=distance[i,:].mean()
        standard=distance[i,:].std()
        #3赛格嘛法则
        # threshold1=average-0.5*standard
        # threshold2=average+0.43*standard
        z=np.abs(norm.ppf(0.2))
        threshold2 =average+(z*standard)
        K = 0
        for j in range(len(distance)):
            if distance[i,j]>threshold2:
                print(distance[i, j])
                K += 1
        print(K)
        for k in range(len(distance) - K):
            index = np.argwhere(distance[i] == np.min(distance[i]))
            distance[i, index] = 10000
        for j in range(len(distance[i])):
            if distance[i, j] != 10000:
                src.append(i)
                dst.append(j)
                # if str(i) in attention_value.keys():
                #     value1 = attention_value[str(i)]
                # value1=attention_value.get(str(i))
                # if str(j) in attention_value:
                #     value2 = attention_value[str(j)]
                # value2 = attention_value.get(str(j))
                # edge_data = [value1, distance[i,j],value2]
                #edge_data = np.mat(edge_data)
                # print("Original dataset = {}*{}".format(edge_data.shape[0], edge_data.shape[1]))
                # pca = PCAcomponent(edge_data, 1)
                # pca.fit()
                # print(pca.low_dataMat)
                # print(pca.variance_ratio)
                # dis.append(distance[i,j])
                # dis.append(edge_data)
                # edge_value[i, j] += edge_data
                # for i in range(len(edge_value)):
                #     for j in range(len(edge_value)):
    data1 = pd.read_csv("/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/preprocess/topo1-testnew_feature_data1", index_col=0)
    a = attention(data1)
    list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    attention_value = dict(zip(list, a.tolist()))
    distance_value = [[] for i in range(len(src))]
    value1=[]
    value2=[]
    for i in range(len(src)):
        s=src[i]
        d=dst[i]
        value1.append(attention_value.get(str(s)))
        # print(attention_value.get(str(s)))
        value2.append(attention_value.get(str(d)))
        for j in range(len(data)):
            cos_value1=data[j, s, 0].reshape(1, -1)   #data[j][s]
            cos_value2=data[j, d, 0].reshape(1, -1)   #data[j][d]
            dis_value =F.cosine_similarity(tensor(cos_value1), tensor(cos_value2), dim=1)
            dis_value = dis_value.item()
            # print(type(dis_value))
            distance_value[i].append([dis_value])

    print(len(data))
    for i in range(len(data)):
        dis = []
        g = dgl.graph((src, dst))
        Feat = th.from_numpy(data[i])
        Feat = Feat.to(th.float32)
        Feat = torch.cat([Feat,a],dim=1)
        # print(type(Feat))
        # print(type(a))
        g.ndata['feature'] = Feat
        # print(type(Feat))
        # print(g.ndata['feature'])
        for j in range(len(src)):
            # edge_data = [value1[j], distance_value[j][i], value2[j]]
            edge_data = [distance_value[j][i]]
            dis.append(edge_data)
            # dis.append(value1[j])
            # dis.append(distance_value[j][i])
            # dis.append(value2[j])
        # print(type(dis))
        dis = np.array(dis)
        # print(type(dis))
        # print(dis)
        dis = torch.from_numpy(dis)
        # print(type(dis))
        g.edata['value'] = dis
        # print(g.edata['value'])
            # for s in range(len(distance)):
            #     for d in range(len(distance[i])):
            #         edge_value=tensor(edge_value)
            #         print(edge_value[s, d])
            #         print(type(edge_value))
            #         if edge_value[s,d]!=[0.,0.,0.]:
            #             g.edata['value']=edge_value[s,d]
            #             print(g.edata['value'])
        g = dgl.add_self_loop(g)
        graph.append(g)
        # nx.draw(g.to_networkx(), with_labels=True)
        # # sleep(5000)
        # plt.show()
    return graph
if __name__ == '__main__':
    #ATTENTION
    # data1 = pd.read_csv("/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/preprocess/graph_feature_data1", index_col=0)
    # data1 = np.array(data1)
    # data1 = Normalization1(data1)
    # line = data1.shape[0]
    # row = data1.shape[1]
    # queries, keys, values, valid_lens=dataprocess(data1,line,row)
    # attention = AdditiveAttention(key_size=line, query_size=row, num_hiddens=line,
    #                               dropout=0.1)
    # attention.eval()  # 开启评估模式
    # a = attention(queries, keys, values, valid_lens)
    # a = a.reshape(11)
    # W_sum = sum(a)
    # a = a / W_sum
    # a=a.view(11,1)
    # print(a)
    # attention(data1)
    # list = ['duration', 'packet_count', 'packet_grow', 'byte_count', 'byte_grow',
    #         'v_byte_flow', 'asym_byte_rate', 'asym_byte_rate_inter', 's_ip_ratio', 'd_ip_ratio', 'ip_tuple']

    # print(attention_value)

    data, label = Feature()
    graph = KNN_graph(data)
    graph_labels = {'label': th.tensor(label)}
    # graphs=[]
    # graphs= graph,graph_labels
    # save_path = "/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/topo2_cos_data2.pkl"
    # with open(save_path, "wb") as f:
    #     pkl.dump(graphs, f)
    # print("Processed Data Saved at {}".format(save_path))
    save_graphs("/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/topo1-sasix_cos_data.bin", graph, graph_labels)
    # g_1, label_1 = load_graphs("/home/ran/Desktop/openrainbow/openrainbow/PT2_GCN/data/graph_flow_data.bin")
    # nx.draw(g_1, with_labels = True)
    # plt.show()
    #
    # g_1, label_1 = load_graphs("./graph.bin",[2])
    # print(len(g_1), type(g_1), len(label_1['label']))
    # print(g_1[0], label_1[0])
    # print(g_1[0].ndata['feature'])
    # nx.draw(g_1[0].to_networkx())
    # plt.show()
