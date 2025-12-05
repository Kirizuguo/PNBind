"""
PNBind - 完全按照论文架构实现
严格遵循 Figure 1 的设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointGNNConv, global_mean_pool as gep


# ====================================================================
# 基础模块（PointGNN 的 MLP）
# ====================================================================

class mlp_h(nn.Module):
    """PointGNN 中的 h_theta MLP"""
    def __init__(self, inputdim=1536, hiddendim=768, outputdim=1536, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputdim, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hiddendim, outputdim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class mlp_f(nn.Module):
    """PointGNN 中的 f_theta MLP"""
    def __init__(self, inputdim=3072, hiddendim=1536, outputdim=1536, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputdim, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hiddendim, outputdim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class mlp_g(nn.Module):
    """PointGNN 中的 g_theta MLP"""
    def __init__(self, inputdim=1536, hiddendim=1536, outputdim=1536, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputdim, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hiddendim, outputdim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.mlp(x)


# ====================================================================
# 🔴 新增：注意力融合模块
# ====================================================================

class AttentionFusion(nn.Module):
    """
    基于注意力的多模态特征融合
    
    替代原来的简单 concat + MLP
    使用 multi-head attention 让不同模态相互交互
    
    参数：
        input_dims: 各模态的输入维度 [semantic_dim, geo_dim, rsa_dim]
        output_dim: 输出维度
        num_heads: 注意力头数
        dropout: dropout 比率
    """
    def __init__(self, input_dims=[512, 1536, 64], output_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_modalities = len(input_dims)
        self.output_dim = output_dim
        
        # 为每个模态学习一个projection到统一维度
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
                nn.LeakyReLU()
            ) for dim in input_dims
        ])
        
        # Multi-head attention用于模态间交互
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # 可学习的query token用于聚合
        self.query_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # 最终输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )
    
    def forward(self, features_list):
        """
        输入：
            features_list: [semantic_feat, geo_feat, rsa_feat]
                semantic_feat: [L, 512]
                geo_feat: [L, 1536]
                rsa_feat: [L, 64]
        
        输出：
            fused: [L, output_dim] (256)
        """
        seq_len = features_list[0].size(0)
        
        # 1. 将所有模态投影到相同维度
        projected = []
        for i, feat in enumerate(features_list):
            proj_feat = self.projections[i](feat)  # [L, output_dim]
            projected.append(proj_feat.unsqueeze(1))  # [L, 1, output_dim]
        
        # 2. 堆叠所有模态: [L, num_modalities, output_dim]
        stacked = torch.cat(projected, dim=1)
        
        # 3. 使用attention聚合跨模态信息
        # Query: 可学习token, Key/Value: 所有模态
        query = self.query_token.expand(seq_len, -1, -1)  # [L, 1, output_dim]
        
        # Attention跨模态
        fused, attn_weights = self.attention(
            query=query,
            key=stacked,
            value=stacked
        )  # fused: [L, 1, output_dim]
        
        fused = fused.squeeze(1)  # [L, output_dim]
        
        # 4. 最终投影
        output = self.output_proj(fused)  # [L, output_dim]
        
        return output


# ====================================================================
# T-Net（空间变换网络）
# ====================================================================

class Tnet(nn.Module):
    """论文中的 T-Net，用于坐标的旋转/平移不变性"""
    def __init__(self, input_dim=3):
        super(Tnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 9)  # 输出 3x3 变换矩阵
        
        self.act = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(1024)
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(256)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = self.act(self.ln3(self.fc3(x)))
        
        # Global pooling
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x = gep(x, batch=batch)
        
        x = self.act(self.ln4(self.fc4(x)))
        x = self.act(self.ln5(self.fc5(x)))
        x = self.fc6(x)
        
        # 加上单位矩阵保证初始化为恒等变换
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(-1)
        x = x + iden
        return x.view(3, 3)


# ====================================================================
# 位置编码
# ====================================================================

def positional_encoding(d_model, max_len=7000):
    """标准的 Transformer 位置编码"""
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros((max_len, d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ====================================================================
# 主模型：PNBind（完全按论文架构）
# ====================================================================

class benchmark(nn.Module):
    """
    PNBind - 核酸结合位点预测模型
    
    架构（严格按照论文 Figure 1）：
    1. 语义分支: ESM3 + HMM + PSSM + DSSP → MLP → [L, 512]
    2. 几何分支: T-Net → PointGNN → [L, 1536]
    3. Late Fusion: 🔴 Attention Fusion → [L, 256]
    4. Decoder: Transformer → Classifier
    
    参数：
        esm_dim: ESM3 输出维度（论文中是 1536）
        num_classes: 分类数（2: 结合/非结合）
        dropout_rate: Dropout 比率
    """
    
    def __init__(self, esm_dim=1536, num_classes=2, dropout_rate=0.2):
        super().__init__()
        
        self.esm_dim = esm_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.LeakyReLU = nn.LeakyReLU()
        
        # ================================================================
        # 分支1：几何分支（Geometric Branch）
        # ================================================================
        
        # T-Net：学习旋转/平移不变的变换
        self.pos_tnet = Tnet(input_dim=3)
        
        # 坐标投影到 ESM 维度空间
        self.pos_proj = nn.Linear(3, esm_dim)
        
        # PointGNN（3层）- 论文中明确提到 3 layers
        self.gnn_layers = nn.ModuleList([
            PointGNNConv(
                mlp_h(esm_dim, esm_dim//2, esm_dim, dropout_rate),
                mlp_f(esm_dim*2, esm_dim, esm_dim, dropout_rate),
                mlp_g(esm_dim, esm_dim, esm_dim, dropout_rate)
            )
            for _ in range(3)
        ])
        
        # ================================================================
        # 分支2：语义分支（Semantic Branch）
        # ================================================================
        
        # 输入: ESM3[1536] + HMM[30] + PSSM[20] + DSSP[13] = 1599
        semantic_input_dim = esm_dim + 30 + 20 + 13  # 1599
        semantic_output_dim = 512
        
        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(768, semantic_output_dim),
            nn.LayerNorm(semantic_output_dim),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU()
        )
        
        # ================================================================
        # 分支3：RSA
        # ================================================================
        
        rsa_output_dim = 64
        self.rsa_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, rsa_output_dim),
            nn.LeakyReLU()
        )
        
        # ================================================================
        # 🔴 Late Fusion Layer (使用注意力融合)
        # ================================================================
        
        fusion_output_dim = 256
        
        # 🔴 新版：使用 AttentionFusion 替代简单的 MLP
        self.fusion_layer = AttentionFusion(
            input_dims=[semantic_output_dim, esm_dim, rsa_output_dim],  # [512, 1536, 64]
            output_dim=fusion_output_dim,  # 256
            num_heads=4,
            dropout=dropout_rate
        )
        
        # ================================================================
        # Transformer Decoder
        # ================================================================
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_output_dim,
            nhead=8,  # 多头注意力
            dim_feedforward=1024,
            dropout=dropout_rate,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        
        # ================================================================
        # 分类头
        # ================================================================
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # ================================================================
        # 初始化
        # ================================================================
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_graph):
        """
        前向传播
        
        输入：
            point_graph: torch_geometric.data.Data 对象
                - x_esm3: [L, 1536] ESM3 特征
                - hmm: [L, 30] HMM 特征
                - pssm: [L, 20] PSSM 特征
                - secondary: [L, 13] DSSP 二级结构
                - asa: [L] 溶剂可及性
                - pos: [L, 3] Cα 坐标
                - edge_index: [2, E] 边索引
        
        输出：
            out: [L, 2] 每个残基的预测分数
            min_len: int 有效残基数量
        """
        
        # ============================================================
        # 提取输入特征
        # ============================================================
        
        edge_index = point_graph.edge_index
        pos = point_graph.pos  # [L, 3]
        
        x_esm = point_graph.x_esm3  # [L, 1536]
        hmm = point_graph.hmm  # [L, 30]
        pssm = point_graph.pssm  # [L, 20]
        secondary = point_graph.secondary  # [L, 13]
        asa = point_graph.asa  # [L]
        
        # ============================================================
        # 长度对齐（防止维度不匹配）
        # ============================================================
        
        min_len = min(
            pos.size(0),
            x_esm.size(0),
            hmm.size(0),
            pssm.size(0),
            secondary.size(0),
            asa.size(0)
        )
        
        pos = pos[:min_len]
        x_esm = x_esm[:min_len]
        hmm = hmm[:min_len]
        pssm = pssm[:min_len]
        secondary = secondary[:min_len]
        asa = asa[:min_len]
        edge_mask = (edge_index[0] < min_len) & (edge_index[1] < min_len)
        edge_index = edge_index[:, edge_mask]
        
        # ============================================================
        # 分支1：几何分支
        # ============================================================
        
        # 1.1 T-Net 空间变换
        T = self.pos_tnet(pos)  # [3, 3]
        pos_transformed = torch.matmul(pos, T)  # [L, 3]
        pos_normalized = F.layer_norm(pos_transformed, (3,))
        
        # 1.2 坐标嵌入
        pos_embed = self.pos_proj(pos_normalized)  # [L, 1536]
        
        # 1.3 PointGNN（3层）
        geo_feat = pos_embed
        for layer in self.gnn_layers:
            geo_feat = self.LeakyReLU(layer(geo_feat, pos_embed, edge_index))
        # geo_feat: [L, 1536]
        
        # ============================================================
        # 分支2：语义分支
        # ============================================================
        
        # 2.1 拼接序列和进化特征
        semantic_input = torch.cat([x_esm, hmm, pssm, secondary], dim=1)  # [L, 1599]
        
        # 2.2 MLP 处理
        semantic_feat = self.semantic_branch(semantic_input)  # [L, 512]
        
        # ============================================================
        # 分支3：RSA
        # ============================================================
        
        rsa_feat = self.rsa_branch(asa.unsqueeze(1))  # [L, 64]
        
        # ============================================================
        # 🔴 Late Fusion (使用注意力融合)
        # ============================================================
        
        # 使用 AttentionFusion 替代简单的 concat + MLP
        fused = self.fusion_layer([semantic_feat, geo_feat, rsa_feat])  # [L, 256]
        
        # ============================================================
        # Transformer Decoder
        # ============================================================
        
        # 添加位置编码
        pe = positional_encoding(fused.size(1), min_len).to(fused.device)
        fused_with_pe = fused + pe[:min_len, :fused.size(1)]
        
        # Transformer 需要 [L, B, D] 格式
        fused_t = fused_with_pe.unsqueeze(1)  # [L, 1, 256]
        decoded = self.decoder(fused_t, fused_t).squeeze(1)  # [L, 256]
        
        # ============================================================
        # 分类
        # ============================================================
        
        out = self.classifier(decoded)  # [L, 2]
        
        return out, min_len


# ====================================================================
# 向后兼容（如果其他代码还在用旧名字）
# ====================================================================

# 这样 train.py 里的 model = benchmark(...) 仍然可以工作