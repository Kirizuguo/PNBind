import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool as gep


class Tnet(nn.Module):
    def __init__(self, input_dim=3):
        super(Tnet,self).__init__()
        self.conv1 = nn.Linear(input_dim, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.LayerNorm(256)

    def forward(self, x, batch):
        """
        # 按照batch扩展维度，无法将图数据（二维tensor）转为点云（三维tensor），
        # 因为图每个数据节点数量不确定，如果转为[batch,node_num,node_feature]，
        # 不能表示为tensor，只能表示为list
        # new_x = torch.tensor([]).to(x.device)
        # temp = torch.tensor([]).to(x.device)
        # flag = 0
        # for i in range(len(batch)):
        #     if batch[i] == flag:
        #         temp = torch.concat((x[i].unsqueeze(0),temp),0)
        #         # temp.append(x[i])
        #     else:
        #         flag = batch[i]
        #         new_x = torch.concat((temp.unsqueeze(0),new_x),0)
        #         temp = torch.tensor([]).to(x.device)
        #         temp = torch.concat((x[i].unsqueeze(0),temp),0)
        #         # flag = batch[i]
        #         # new_x.append(temp)
        #         # temp = []
        #         # temp.append(x[i])
        # new_x = torch.concat((temp.unsqueeze(0),new_x),0)
        # # new_x.append(temp)
        # x = torch.tensor(new_x).permute(0, 2, 1)
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = torch.max(x, 1, keepdim=True)[0]
        x = gep(x, batch=batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.bn5(self.fc2(x)))
        # x = F.relu(self.fc1(x))  # 先不归一化
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            1, 1)  # 因为batch是1，所以只重复一次
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(3, 3)
        return x


class Gcn_Attention(nn.Module):
    def __init__(self, act, input_dim=3, output_dim=64):
        super(Gcn_Attention, self).__init__()
        self.li1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.gcn1 = GCNConv(in_channels=output_dim, out_channels=output_dim)
        self.act = act
        self.norm1d = nn.LayerNorm(output_dim*2)
    
    def forward(self, x, edge_index):
        node_features = self.act(self.li1(x))
        node_features2 = self.act(self.gcn1(node_features, edge_index))
        node_features = self.norm1d(torch.concat((node_features, node_features2), 1))
        return node_features
    

class PCM_noConv(nn.Module):
    def __init__(self, act=nn.LeakyReLU(), input_dim=3, hidden_dim=64, output_dim=4):
        super(PCM_noConv, self).__init__()
        self.act = act
        self.tnet = Tnet(input_dim=3)
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.gcn_a1 = Gcn_Attention(act=self.act, input_dim=hidden_dim, output_dim=hidden_dim)
        self.conv2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.gcn_a2 = Gcn_Attention(act=self.act, input_dim=hidden_dim*2, output_dim=hidden_dim*2)
        self.conv3 = nn.Linear(hidden_dim*2*2+hidden_dim, hidden_dim*4)
        self.lin_dim = hidden_dim*4
        # self.conv4 = nn.Conv1d(hidden_dim*2, output_dim, 1)
        self.li1 = nn.Linear(hidden_dim*2*2+hidden_dim, hidden_dim*3)
        self.li2 = nn.Linear(hidden_dim*3, hidden_dim)
        self.li3 = nn.Linear(hidden_dim, output_dim)

        # self.li1 = nn.Linear()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, point_graph):
        input_point, batch = point_graph.x, point_graph.batch
        x = torch.mm(input_point, self.tnet(input_point, batch))
        x_global = self.act(self.conv1(x))
        x = self.gcn_a1(x_global, point_graph.edge_index)  # 相对全局的信息
        x = self.act(self.conv2(x))
        x = self.gcn_a2(x, point_graph.edge_index)

        x = torch.concat((x,x_global),1)
        x = self.act(self.conv3(x))
        x = gep(x, batch=batch)
        # x = x.view(-1, self.lin_dim)
        # x = torch.concat((x_global,x),0)
        x = self.concate_x_xglobal(x, x_global, batch)
        # x = self.conv4(x)
        x = self.act(self.li1(x))
        x = self.act(self.li2(x))
        x = self.act(self.li3(x))
        return x

    def concate_x_xglobal(self, x, x_global, batch):
        # 输入只能是1个batch
        x = x.repeat(x_global.shape[0],1)

        return torch.concat((x, x_global), 1)








