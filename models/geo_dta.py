# 基于gnn_node.py设计的有边信息的模型,这里的图卷积聚合是自己写的，基于GCNConv
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv, Sequential, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import inits, MessagePassing
from torch_geometric.utils import add_self_loops


"""
# class Linear(nn.Module):

#     def __init__(self, in_channels, out_channels, bias=True,
#                  weight_initializer='glorot',
#                  bias_initializer='zeros'):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight_initializer = weight_initializer
#         self.bias_initializer = bias_initializer

#         assert in_channels > 0
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.in_channels > 0:
#             if self.weight_initializer == 'glorot':
#                 inits.glorot(self.weight)
#             elif self.weight_initializer == 'glorot_orthogonal':
#                 inits.glorot_orthogonal(self.weight, scale=2.0)
#             elif self.weight_initializer == 'uniform':
#                 bound = 1.0 / math.sqrt(self.weight.size(-1))
#                 torch.nn.init.uniform_(self.weight.data, -bound, bound)
#             elif self.weight_initializer == 'kaiming_uniform':
#                 inits.kaiming_uniform(self.weight, fan=self.in_channels,
#                                       a=math.sqrt(5))
#             elif self.weight_initializer == 'zeros':
#                 inits.zeros(self.weight)
#             elif self.weight_initializer is None:
#                 inits.kaiming_uniform(self.weight, fan=self.in_channels,
#                                       a=math.sqrt(5))
#             else:
#                 raise RuntimeError(
#                     f"Linear layer weight initializer "
#                     f"'{self.weight_initializer}' is not supported")

#         if self.in_channels > 0 and self.bias is not None:
#             if self.bias_initializer == 'zeros':
#                 inits.zeros(self.bias)
#             elif self.bias_initializer is None:
#                 inits.uniform(self.in_channels, self.bias)
#             else:
#                 raise RuntimeError(
#                     f"Linear layer bias initializer "
#                     f"'{self.bias_initializer}' is not supported")

#     def forward(self, x):
#         """"""
#         return F.linear(x, self.weight, self.bias)


# class SelfAttention(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout, device):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads

#         assert hid_dim % n_heads == 0

#         self.w_q = nn.Linear(hid_dim, hid_dim)
#         self.w_k = nn.Linear(hid_dim, hid_dim)
#         self.w_v = nn.Linear(hid_dim, hid_dim)

#         self.fc = nn.Linear(hid_dim, hid_dim)

#         self.do = nn.Dropout(dropout)

#         self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

#     def forward(self, query, key, value, mol_batch, pro_batch, mask=None):
#         # bsz = query.shape[0]

#         # query = key = value [batch size, sent len, hid dim]

#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)

#         # Q, K, V = [batch size, sent len, hid dim]

#         Q = Q.view(-1, self.n_heads, self.hid_dim // self.n_heads).permute(1, 0, 2)
#         K = K.view(-1, self.n_heads, self.hid_dim // self.n_heads).permute(1, 0, 2)
#         V = V.view(-1, self.n_heads, self.hid_dim // self.n_heads).permute(1, 0, 2)
        
#         # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
#         # Q = [batch size, n heads, sent len_q, hid dim // n heads]
#         unique_values_mol, batch_count_mol = torch.unique(mol_batch, return_counts=True)
#         unique_values_pro, batch_count_pro = torch.unique(pro_batch, return_counts=True)
#         if len(unique_values_mol) != len(unique_values_pro):
#             raise ValueError("mol_batch should be same as pro_batch")
#         K = K.permute(0, 2, 1)
#         count_where_mol = 0
#         count_where_pro = 0
#         x = torch.empty_like(Q)
#         for i in unique_values_pro:
#             energy = torch.matmul(Q[:, count_where_mol:count_where_mol+batch_count_mol[i],:], K[:, :, count_where_pro:count_where_pro+batch_count_pro[i]]) / self.scale
#         # energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

#         # energy = [batch size, n heads, sent len_Q, sent len_K]
#             if mask is not None:
#                 energy = energy.masked_fill(mask == 0, -1e10)

#             attention = F.softmax(energy, dim=-1)  # 这里为什么参数是-1呢？

#             # attention = [batch size, n heads, sent len_Q, sent len_K]

#             x[:, count_where_mol:count_where_mol+batch_count_mol[i],:] = torch.matmul(attention, V[:, count_where_pro:count_where_pro+batch_count_pro[i],:])  # 这里没有转置，说明先验是len(v) == len(k)

#             # x = [batch size, n heads, sent len_Q, hid dim // n heads]
#             count_where_mol += batch_count_mol[i]
#             count_where_pro += batch_count_pro[i]

#         x = x.permute(1,0,2).contiguous()

#         # x = [batch size, sent len_Q, n heads, hid dim // n heads]

#         x = x.view(-1, self.n_heads * (self.hid_dim // self.n_heads))

#             # x = [batch size, src sent len_Q, hid dim]

#         x = self.fc(x)

#         # x = [batch size, sent len_Q, hid dim]

#         return x


# class DecoderLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
#         super().__init__()

#         self.ln = nn.LayerNorm(hid_dim)
#         self.sa = self_attention(hid_dim, n_heads, dropout, device)
#         self.ea = self_attention(hid_dim, n_heads, dropout, device)
#         self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
#         self.do = nn.Dropout(dropout)

#     def forward(self, trg, src, trg_mask=None, src_mask=None):
#         # trg = [batch_size, compound len, atom_dim]
#         # src = [batch_size, protein len, hid_dim] # encoder output
#         # trg_mask = [batch size, compound sent len]
#         # src_mask = [batch size, protein len]

#         # trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))   # 这一层是用于target的自监督学习

#         trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))  # 将source和target结合起来

#         trg = self.ln(trg + self.do(self.pf(trg)))

#         return trg
"""

class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.aggr = 'maen'
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        
        return edge_weight * x_j


 # “Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification”transformer图卷积


class protein_gnn(nn.Module):
    def __init__(self, act, len_feature_edge, hidden_channels=128, output_channels=128, dropout=0.2):
        super(protein_gnn, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.act = act
        self.lin_x1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_x2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_edge1 = nn.Linear(len_feature_edge, hidden_channels)
        self.lin_edge2 = nn.Linear(hidden_channels, hidden_channels)
        self.final = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, t_feature_edge, t_edge_index):
        x_lin_1 = self.act(self.lin_x1(x))
        x_lin_2 = self.act(self.lin_x2(x))

        feature1 = self.lin_edge1(t_feature_edge)
        h1 = self.conv1(x_lin_1, t_edge_index, feature1)
        h1 = self.act(self.lin_edge2(h1))
        h1 = self.dropout(h1)

        h = h1 + x_lin_2
        h = self.final(h)
        return h


class protein_conv(nn.Module):
    def __init__(self, len_pc_feature, len_feature_edge, hidden_channels, act, dropout, num_blocks=1):
        super(protein_conv, self).__init__()
        self.lin_pc = nn.Linear(len_pc_feature, hidden_channels)
        # self.len_feature_edge
        self.lin_x = nn.Linear(hidden_channels, hidden_channels)
        self.lin_edge_feature = nn.Linear(len_feature_edge, hidden_channels)
        self.act = act
        self.protein_gnn_blocks = nn.ModuleList([
            protein_gnn(act=act,
                        len_feature_edge=len_feature_edge,
                        hidden_channels = hidden_channels, 
                        output_channels=hidden_channels,
                        dropout=dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # 对GCN的尝试
        # 自定义的transformer图卷积
        def define_layers(hidden_channels):
            gcn_layer_sizes = [hidden_channels,hidden_channels]
            layers = []
            for i in range(len(gcn_layer_sizes) - 1):            
                layers.append((
                    TransformerConv(
                        gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=hidden_channels),
                    'x, edge_index, edge_attr -> x'
                ))
                layers.append(nn.LeakyReLU())
                return layers

        # self.gcn_protein0 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channels))
        # self.gcn_protein1 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channels))
        # self.gcn_protein2 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channels))
        
        self.gcn_protein0 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        self.gcn_protein1 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        self.gcn_protein2 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        
    def forward(self, data_pro):
        # target_x, target_edge_weights, target_edge_index, target_batch = data_pro.x[:,:-6], data_pro.edge_weights, data_pro.edge_index, data_pro.batch
        t_pc_feature, t_feature_edge, t_edge_index= data_pro.x, data_pro.t_feature_edge, data_pro.edge_index
        
        # t_edge_index,_ = add_self_loops(t_edge_index, num_nodes=len(data_pro))
        t_batch = data_pro.batch
        # t_pc_feature = self.act(self.lin_pc(t_pc_feature.to(torch.float32)))
        # pro_encode = self.act(self.lin_x(t_pc_feature))  # 这里按理说应该都是在最初特征上分别做两次线性变换，而不是在一次变换的基础上再做一次变换
        # for protein_gnn_block in self.protein_gnn_blocks:
        #     pro_encode = protein_gnn_block(pro_encode, t_feature_edge, t_edge_index) + pro_encode
        
        # pro_encode = torch.cat((t_pc_feature, pro_encode),1)

        # transformergcn尝试
        # edge_feature = self.act(self.lin_edge_feature(torch.cat((t_pos_embeding, t_feature_edge),1)))
        t_pc_feature = self.act(self.lin_pc(t_pc_feature.to(torch.float32)))
        pro_encode = self.act(self.gcn_protein0(t_pc_feature,t_edge_index)) + t_pc_feature
        pro_encode = self.act(self.gcn_protein1(pro_encode,t_edge_index)) + pro_encode
        pro_encode = self.act(self.gcn_protein2(pro_encode,t_edge_index)) + pro_encode
        pro_encode = torch.cat((t_pc_feature, pro_encode),1)

        return pro_encode

# 原本药物部分的特征维度：num_features_mol=78, num_edge_feature_mol=1
# 蛋白质33
# GCN based model
class GCN_Edge(nn.Module):
    def __init__(self, 
                 len_pc_feature=4096, len_feature_edge=16, num_blocks=3,
                 hidden_channels=256, dropout=0,
                 n_output=2):
        super(GCN_Edge, self).__init__()

        self.act = nn.LeakyReLU()
        # 蛋白质部分
        self.pronet = protein_conv(len_pc_feature=len_pc_feature, 
                                   len_feature_edge=len_feature_edge,
                                   hidden_channels=hidden_channels,
                                   num_blocks=num_blocks, 
                                   dropout=dropout, 
                                   act=self.act)
        self.lin_pro0 = nn.Linear(len_pc_feature, hidden_channels*2)
        self.lin_pro1 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.lin_pro2 = nn.Linear(hidden_channels*4, hidden_channels*2)
        
        # 拼接
        self.fc1 = nn.Linear(hidden_channels*2, hidden_channels)
        self.final = nn.Linear(hidden_channels, n_output)
        self.dropout = nn.Dropout(dropout)
        self.layernorm_drug = nn.LayerNorm(normalized_shape=hidden_channels*2, elementwise_affine=True)
        self.layernorm_protein = nn.LayerNorm(normalized_shape=hidden_channels*2, elementwise_affine=True)
        self.layernorm = nn.LayerNorm(normalized_shape=hidden_channels*4, elementwise_affine=True)
        self.norm1d = nn.BatchNorm1d(hidden_channels*4)
        # self.norm1d = nn.BatchNorm1d(hidden_channels*4)

    def forward(self, data_pro):
        # pro
        # protein_encode = self.layernorm_protein(self.pronet(data_pro))
        protein_encode = self.layernorm_protein(self.lin_pro0(data_pro.x.to(torch.float32)))
        protein_encode = self.act(self.lin_pro1(protein_encode))
        protein_encode = self.act(self.lin_pro2(protein_encode))

        # add some dense layers
        xc = self.act(self.fc1(protein_encode))
        xc = self.dropout(xc)
        out = self.final(xc)

        return out

