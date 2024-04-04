# -*- coding: utf-8 -*-
from dataset import MyDataset
import dgl
import torch
import numpy as np
import dgl.nn as dglnn
from torch.utils.data import DataLoader, Subset
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from confusion import plot_confusion_matrix
from torch.utils.data import random_split



class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,dropout1=0.7):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.SAGEConv(
            in_feats=in_dim, out_feats=hidden_dim, aggregator_type='lstm')
        self.conv5 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type='lstm')
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=32)
        self.classify =nn.Sequential(nn.Dropout(p=dropout1), nn.Linear(32, n_classes))
        # self.classify = nn.Sequential(nn.Linear(32, n_classes))
        # self.classify = nn.Linear(hidden_dim, n_classes)
        # self.classify = nn.Linear(hidden_dim, n_classes)



    def forward(self, g):
        h = g.ndata['feature'].float()
        #new model
        # h = g.edata['value'].float()
        h = F.relu(self.conv4(g, h))
        # h = torch.tanh(self.conv2(g, h))
        h = F.relu(self.conv2(g, h))
        #TGCN
        # h = F.relu(self.conv1(g, h))
        # h = torch.tanh(self.conv2(g, h))
        list1 = list(h.size())
        h = h.view(1, list1[0], list1[1])
        out, hidden = self.gru(h)
        list2 = list(out.size())
        out = out.view(list2[1], 32)
        g.ndata['feature'] = out
        hg = dgl.mean_nodes(g, 'feature') # 图的特征是所有节点特征的均值
        y = self.classify(hg)
        return y



    # return A(graph, feat, weight, ntype=ntype, op='mean')

# SAGEConv can be applied on homogeneous graph and unidirectional
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(SAGE, self).__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.classify = nn.Linear(hid_feats, out_feats)

    def forward(self, graph):
        # 输入是节点的特征
        h = graph.ndata['feature'].float()
        # h = graph.edata['value'].float()
        h = self.conv1(graph, h)
        h = F.relu(h)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        graph.ndata['feature'] = h
        hg = dgl.mean_nodes(graph, 'feature')
        h = self.classify(hg)
        return h

a = '40'
# DataSet = MyDataset('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/'+a+'/forward_data.bin')
DataSet = MyDataset('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/graph_data(safive).bin')
# DataSet1 = np.load('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/dataset_with_labels.npy', allow_pickle=True)
# DataSet = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/data/90/knn_flow_data.bin')
length = len(DataSet)
# length = len(DataSet1)
# print(length)
trainset = Subset(DataSet, range(0, int(0.6*length))) #1,4001;4002,6370
testset = Subset(DataSet, range(int(0.6*length), length))

# train_size = int(length*0.6)
# test_size = length-train_size
# trainset, testset = random_split(dataset=DATA, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(10))
# print(len(testset))
# print(len(trainset))
# 训练模型


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    # print(graphs)
    batched_graph = dgl.batch(graphs)
    # print(batched_graph)
    return batched_graph, torch.tensor(labels)


def Train():
    # trainset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/data_set.bin')
    # trainset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/forward_data.bin', 2000, 0)
    # testset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/forward_data.bin', 1200, 1)
    #forward_data
    # print(0.6*length)

    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
#10 batch_size=64, 30 batch_size=64, 50 batch_size=32, 70 batch_size=32 90 batch_size=64
    a = input('Select Model (0-Gcn, 1-SAGEGcn):')
    #forward_data
    if a == '0':
        model = GCN(12, 128, 2)
    if a == '1':
        model = SAGE(2, 128, 2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # metric: patience is the most epochs about the loss not reducing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    #Trainning
    model.train()
    epoch_losses = []
    Acc = []
    lr = []
    max_acc = 0
    for epoch in range(100): #forward_data 250
        epoch_loss = 0
        correct = 0
        L = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            label = label.type(torch.LongTensor)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().item()
            predicted = torch.max(prediction.data, 1)[1]
            correct += (predicted == label).sum()
            L += len(label)
        epoch_loss /= (iter+1)
        accuracy = float(correct*100)/L
        scheduler.step(epoch_loss)
        new_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # Test
        test_X, test_Y = map(list, zip(*testset))
        test_bg = dgl.batch(test_X)
        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        probs_Y = torch.softmax(model(test_bg), 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)

        # print(len(test_X), model(test_bg).size())
        # print(test_Y, argmax_Y)
        test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
        if max_acc < test_acc:
            max_acc = test_acc
            if a == '0':
                torch.save(model, 'usage.pkl')
            if a == '1':
                torch.save(model, 'sage.pkl')

        print('Epoch {}, loss {:.4f}, Train Acc {:.4f}, Test Acc {:.4f}, Learning Rate {:.10f}'.format(epoch+1,
                                                                                                       epoch_loss, accuracy,
                                                                                                       test_acc,
                                                                                                       new_lr))
        epoch_losses.append(epoch_loss)
        Acc.append(accuracy)
        lr.append(new_lr)



    plt.title('cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    plt.show()
    plt.title('Accuracy')
    plt.plot(Acc)
    plt.show()
    # plt.title('Learning Rate')
    # plt.plot(lr)
    # plt.show()


    # save model, 2 methods
    # torch.save(model.state_dict(), 'gcn.pkl')
    # if a=='0':
    #     torch.save(model, 'gcn.pkl')
    # if a=='1':
    #     torch.save(model, 'sage.pkl')
    return model, a


def Test(Model, flag):
    #Testing
    # testset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/data_set.bin')
    # testset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/data_set.bin', 750, seed+1)
    # testset = Subset(DataSet, range(5001, 7000))
    # testset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/trans/forward_data.bin', 1200, 1)
    # testset = Subset(DataSet, range(3001, 5001))
    # testset = Subset(DataSet, range(int(0.6*length), length))

    # tstset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/data/50/forward_data.bin',
    #                     test_size=0.4)

    # 2 methods about loading torch gcn model
    # model = Classifier(4, 16, testset.num_labels)
    # model.load_state_dict(torch.load('gcn.pkl'))
    # model = Model
    if flag == '0':
        model = torch.load('usage.pkl')
    if flag == '1':
        model = torch.load('sage.pkl')

    # model = torch.load('gcn.pkl')
    model.eval()

    # Convert a list of tuples to two lists
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    print('Confusion matrix:')
    print(test_Y.size(), argmax_Y.size())
    cm = confusion_matrix(test_Y, argmax_Y)
    classes = ['Normal', 'SA']
    plot_confusion_matrix(cm, classes)

    test_Y = test_Y.numpy()
    Y_pr = probs_Y.data[:, 1].numpy()
    # print(Y_pr)
    L = np.concatenate((test_Y.reshape((len(test_Y), 1)), Y_pr.reshape((len(Y_pr), 1))), axis=1)
    # np.savetxt('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/' + a + '/gcn_label.txt', L,
    #            fmt='%d %.3f')
    argmax_Y = argmax_Y.numpy()
    TN = np.sum((test_Y == 0) & (argmax_Y == 0))
    FP = np.sum((test_Y == 0) & (argmax_Y == 1))
    FN = np.sum((test_Y == 1) & (argmax_Y == 0))
    TP = np.sum((test_Y == 1) & (argmax_Y == 1))
    accuracy = (TP + TN) / (TN + FP + FN + TP)
    precision = TP / (FP + TP)
    recall = TP / (FN + TP)
    f_1 = f1_score(test_Y, argmax_Y)
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    L = np.concatenate((test_Y.reshape((len(test_Y), 1)), Y_pr.reshape((len(Y_pr), 1))), axis=1)
    np.savetxt('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/ls-usage.txt', L,
               fmt='%d %.3f')
    print(
        'TPR {:.7f}, TNR {:.7f}, Acc {:.7f}, Precision {:.7f}, Recall {:.7f} F1 {:.7f}'.format(tpr, tnr, accuracy,
                                                                                               precision,
                                                                                               recall, f_1))

if __name__=='__main__':
    #model, flag = Train()
    # print(model)
    # model = torch.load('usage.pkl')
    # flag = 00
    Test("0", "0")
