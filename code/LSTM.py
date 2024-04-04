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
import random

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout1 = 0.7):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, 8)
        self.GRU = nn.LSTM(input_size=in_dim,hidden_size=hidden_dim,batch_first=True)
        # self.GRU1 = nn.LSTM(input_size=hidden_dim,hidden_size=hidden_dim,batch_first=True)

        # self.fu = nn.Linear(hidden_dim, 64)
        # self.fc = nn.Linear(64, 2)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Dropout(p=dropout1), nn.Linear(64, 2))

        # # 定义注意力层
        # self.attention = nn.Sequential(
        #     nn.Linear(2, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, 2)
        # )


        self.classify = nn.Linear(8, n_classes)
        self.classify1 = nn.Sequential(nn.Dropout(p=dropout1), nn.Linear(8, n_classes))
        # self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feature'].float()

        size = h.shape[0]
        size2 = h.shape[1]
        hgru = h.view(int(size/11),11,size2)
        h1,_ = self.GRU(hgru)
        # h1,_ = self.GRU1(h1)
        # h3 = self.fu(h1[:,-1,:])
        # out = self.fc(h3)
        out = self.ff(h1[:,-1,:])
        # att = F.softmax(self.attention(out), dim=1)
        # out_att = att*out
        h = F.relu(self.conv1(g, h))
        # h = torch.tanh(self.conv1(g, h))

        h = torch.tanh(self.conv2(g, h))

        # h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        # h = torch.tanh(self.conv3(g, h))

        g.ndata['feature'] = h ## 节点特征经过两层卷积的输出
        hg = dgl.mean_nodes(g, 'feature') # 图的特征是所有节点特征的均值
        y = self.classify1(hg)
        out1 = y + out
        return out1


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
        h = self.conv1(graph, h)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        graph.ndata['feature'] = h
        hg = dgl.mean_nodes(graph, 'feature')
        h = self.classify(hg)
        return h


# seed = random.randint(3,8)
# print('seed %d'%seed)
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
# 训练模型


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
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
    # data_loader1 = DataLoader(trainset1, batch_size=32, shuffle=True, collate_fn=collate)
    # print('Training')
#10 batch_size=64, 30 batch_size=64, 50 batch_size=32, 70 batch_size=32 90 batch_size=64
    a = input('Select Model (0-Gcn, 1-SAGEGcn):')
    #forward_data
    if a == '0':
        model = GCN(12, 128, 2)
    if a == '1':
        model = SAGE(2, 256, 2)

    # #knn_flow_data
    # if a=='0':
    #     model = GCN(1, 128, 2)
    # if a=='1':
    #     model = SAGE(1, 256, 2)
    # model = Classifier1(4, 256, 2)
    # model = Classifier(4, 16, trainset.num_labels)
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
            # print(len(bg))
            # print(bg)
            prediction = model(bg)
            label = label.type(torch.LongTensor)
            loss = loss_func(prediction, label)
            # print(loss, label)
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
        test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
        if max_acc < test_acc:
            max_acc = test_acc
            if a == '0':
                torch.save(model, 'lstm.pkl')
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

    # testset = MyDataset('/home/rly/work/openrainbow_r/openrainbow/openrainbow/GCN/data/50/forward_data.bin',
    #                     test_size=0.4)

    # 2 methods about loading torch gcn model
    # model = Classifier(4, 16, testset.num_labels)
    # model.load_state_dict(torch.load('gcn.pkl'))
    # model = Model
    if flag == '0':
        model = torch.load('lstm.pkl')
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
    np.savetxt('/home/ran/Desktop/openrainbow/openrainbow/GCN/data/LSTM-gu-gcn.txt', L,
               fmt='%d %.3f')
    print(
        'TPR {:.6f}, TNR {:.6f}, Acc {:.6f}, Precision {:.6f}, Recall {:.6f} F1 {:.6f}'.format(tpr, tnr, accuracy,
                                                                                               precision,
                                                                                               recall, f_1))

if __name__=='__main__':
    # model, flag = Train()
    Test("0", "0")