"""
PNBind - 局部Cross-Attention版本
通过k-NN约束attention感受野，保留几何局部性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import PointGNNConv, global_mean_pool as gep

# ====================================================================
# 基础模块（复用）
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
# 🔴 核心改进：局部Cross-Attention with Gated Residual
# ====================================================================

class LocalCrossModalAttention(nn.Module):
    """
    局部化Cross-Attention融合
    
    关键改进：
    1. k-NN mask: 只在局部邻域内计算attention
    2. Gated Residual: 可学习的残差权重
    3. 保留几何局部性
    """
    def __init__(self, geo_dim=1536, sem_dim=512, rsa_dim=64, 
                 output_dim=256, num_heads=8, k_neighbors=10, dropout=0.2):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.k = k_neighbors
        
        # 投影层
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.rsa_proj = nn.Sequential(
            nn.Linear(rsa_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention components
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        # 🔴 Gated Residual 可学习权重
        self.alpha_geo = nn.Parameter(torch.tensor(1.0))
        self.beta_geo = nn.Parameter(torch.tensor(0.5))
        self.alpha_sem = nn.Parameter(torch.tensor(1.0))
        self.beta_sem = nn.Parameter(torch.tensor(0.5))
        
        # RSA独立处理
        self.rsa_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU()
        )
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU()
        )
        
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def build_knn_mask(self, pos, k):
        """
        构建k-NN邻域mask
        
        Args:
            pos: [L, 3] 3D坐标
            k: int, 邻居数量
        
        Returns:
            mask: [L, L] bool tensor, True表示可以attend
        """
        L = pos.size(0)
        
        # 计算欧氏距离
        dist = torch.cdist(pos, pos)  # [L, L]
        
        # 找到最近的k个邻居（包括自己）
        _, indices = torch.topk(dist, min(k, L), largest=False, dim=1)
        
        # 构建mask
        mask = torch.zeros(L, L, dtype=torch.bool, device=pos.device)
        mask.scatter_(1, indices, True)
        
        return mask
    
    def local_attention(self, query, key, value, mask):
        """
        局部masked attention
        
        Args:
            query, key, value: [L, num_heads, head_dim]
            mask: [L, L] bool mask
        
        Returns:
            output: [L, num_heads, head_dim]
        """
        # 🔴 转换维度: [L, num_heads, head_dim] -> [num_heads, L, head_dim]
        query = query.transpose(0, 1)  # [num_heads, L, head_dim]
        key = key.transpose(0, 1)      # [num_heads, L, head_dim]
        value = value.transpose(0, 1)  # [num_heads, L, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [num_heads, L, L]
        scores = scores / math.sqrt(self.head_dim)
        
        # 🔴 应用k-NN mask（远距离设为-inf）
        if mask is not None:
            # mask: [L, L] -> [1, L, L] -> broadcast to [num_heads, L, L]
            mask = mask.unsqueeze(0)  # [1, L, L]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, value)  # [num_heads, L, head_dim]
        
        # 🔴 转换回来: [num_heads, L, head_dim] -> [L, num_heads, head_dim]
        output = output.transpose(0, 1)
        
        return output
    
    def forward(self, geo_feat, sem_feat, rsa_feat, pos):
        """
        前向传播
        
        Args:
            geo_feat: [L, 1536] 几何特征
            sem_feat: [L, 512] 语义特征
            rsa_feat: [L, 64] RSA特征
            pos: [L, 3] 3D坐标（用于构建k-NN）
        
        Returns:
            fused: [L, 256] 融合特征
        """
        L = geo_feat.size(0)
        
        # 投影到统一维度
        geo_proj = self.geo_proj(geo_feat)  # [L, 256]
        sem_proj = self.sem_proj(sem_feat)  # [L, 256]
        rsa_proj = self.rsa_proj(rsa_feat)  # [L, 256]
        
        # 🔴 构建k-NN mask（基于3D坐标）
        knn_mask = self.build_knn_mask(pos, self.k)  # [L, L]
        
        # ============================================================
        # 几何 attend to 语义（局部）
        # ============================================================
        geo_q = self.q_proj(geo_proj).view(L, self.num_heads, self.head_dim)
        sem_k = self.k_proj(sem_proj).view(L, self.num_heads, self.head_dim)
        sem_v = self.v_proj(sem_proj).view(L, self.num_heads, self.head_dim)
        
        geo_attn_out = self.local_attention(geo_q, sem_k, sem_v, knn_mask)
        geo_attn_out = geo_attn_out.reshape(L, self.output_dim)
        geo_attn_out = self.out_proj(geo_attn_out)
        
        # 🔴 Gated Residual融合
        geo_output = self.layer_norm1(
            self.alpha_geo * geo_proj + self.beta_geo * geo_attn_out
        )
        
        # ============================================================
        # 语义 attend to 几何（局部）
        # ============================================================
        sem_q = self.q_proj(sem_proj).view(L, self.num_heads, self.head_dim)
        geo_k = self.k_proj(geo_proj).view(L, self.num_heads, self.head_dim)
        geo_v = self.v_proj(geo_proj).view(L, self.num_heads, self.head_dim)
        
        sem_attn_out = self.local_attention(sem_q, geo_k, geo_v, knn_mask)
        sem_attn_out = sem_attn_out.reshape(L, self.output_dim)
        sem_attn_out = self.out_proj(sem_attn_out)
        
        # 🔴 Gated Residual融合
        sem_output = self.layer_norm2(
            self.alpha_sem * sem_proj + self.beta_sem * sem_attn_out
        )
        
        # ============================================================
        # RSA单独处理
        # ============================================================
        rsa_output = self.rsa_transform(rsa_proj)
        
        # ============================================================
        # 最终融合三个模态
        # ============================================================
        combined = torch.cat([geo_output, sem_output, rsa_output], dim=1)
        fused = self.fusion(combined)
        
        return fused
# ====================================================================
# T-Net
# ====================================================================

class Tnet(nn.Module):
    """T-Net用于坐标的旋转/平移不变性"""
    def __init__(self, input_dim=3):
        super(Tnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 9)
        
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
        
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x = gep(x, batch=batch)
        
        x = self.act(self.ln4(self.fc4(x)))
        x = self.act(self.ln5(self.fc5(x)))
        x = self.fc6(x)
        
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(-1)
        x = x + iden
        return x.view(3, 3)


# ====================================================================
# 位置编码
# ====================================================================

def positional_encoding(d_model, max_len=7000):
    """标准Transformer位置编码"""
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros((max_len, d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ====================================================================
# 主模型
# ====================================================================

class benchmark(nn.Module):
    """
    PNBind - 局部Cross-Attention版本
    
    关键改进：
    1. 局部k-NN约束的Cross-Attention
    2. 可学习的Gated Residual
    3. 保留GNN的局部几何特征
    """
    
    def __init__(self, esm_dim=1536, num_classes=2, dropout_rate=0.2, k_neighbors=10):
        super().__init__()
        
        self.esm_dim = esm_dim
        self.k_neighbors = k_neighbors
        self.dropout = nn.Dropout(dropout_rate)
        self.LeakyReLU = nn.LeakyReLU()
        
        # ================================================================
        # 几何分支
        # ================================================================
        self.pos_tnet = Tnet(input_dim=3)
        self.pos_proj = nn.Linear(3, esm_dim)
        
        self.gnn_layers = nn.ModuleList([
            PointGNNConv(
                mlp_h(esm_dim, esm_dim//2, esm_dim, dropout_rate),
                mlp_f(esm_dim*2, esm_dim, esm_dim, dropout_rate),
                mlp_g(esm_dim, esm_dim, esm_dim, dropout_rate)
            )
            for _ in range(3)
        ])
        
        # ================================================================
        # 语义分支
        # ================================================================
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
        # RSA分支
        # ================================================================
        rsa_output_dim = 64
        self.rsa_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, rsa_output_dim),
            nn.LayerNorm(rsa_output_dim),
            nn.LeakyReLU()
        )
        
        # ================================================================
        # 🔴 局部Cross-Attention融合层
        # ================================================================
        fusion_output_dim = 256
        
        self.fusion_layer = LocalCrossModalAttention(
            geo_dim=esm_dim,
            sem_dim=semantic_output_dim,
            rsa_dim=rsa_output_dim,
            output_dim=fusion_output_dim,
            num_heads=8,
            k_neighbors=k_neighbors,
            dropout=dropout_rate
        )
        
        # ================================================================
        # Transformer Decoder
        # ================================================================
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_output_dim,
            nhead=8,
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
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_graph):
        """前向传播"""
        # 提取输入
        edge_index = point_graph.edge_index
        pos = point_graph.pos
        x_esm = point_graph.x_esm3
        hmm = point_graph.hmm
        pssm = point_graph.pssm
        secondary = point_graph.secondary
        asa = point_graph.asa
        
        # 长度对齐
        min_len = min(
            pos.size(0), x_esm.size(0), hmm.size(0),
            pssm.size(0), secondary.size(0), asa.size(0)
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
        # 几何分支
        # ============================================================
        # 坐标标准化
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_normalized = pos_centered / (pos_centered.std(dim=0, keepdim=True) + 1e-6)
        
        # T-Net对齐
        T = self.pos_tnet(pos_normalized)
        pos_transformed = torch.matmul(pos_normalized, T)
        
        # 投影 + GNN
        pos_embed = self.pos_proj(F.layer_norm(pos_transformed, (3,)))
        
        geo_feat = pos_embed
        for layer in self.gnn_layers:
            geo_feat = self.LeakyReLU(layer(geo_feat, pos_embed, edge_index))
        
        # ============================================================
        # 语义分支
        # ============================================================
        semantic_input = torch.cat([x_esm, hmm, pssm, secondary], dim=1)
        semantic_feat = self.semantic_branch(semantic_input)
        
        # ============================================================
        # RSA分支
        # ============================================================
        rsa_feat = self.rsa_branch(asa.unsqueeze(1))
        
        # ============================================================
        # 🔴 局部Cross-Attention融合（传入pos用于k-NN）
        # ============================================================
        fused = self.fusion_layer(geo_feat, semantic_feat, rsa_feat, pos_normalized)
        
        # ============================================================
        # Transformer Decoder
        # ============================================================
        pe = positional_encoding(fused.size(1), min_len).to(fused.device)
        fused_with_pe = fused + pe[:min_len, :fused.size(1)]
        
        fused_t = fused_with_pe.unsqueeze(1)
        decoded = self.decoder(fused_t, fused_t).squeeze(1)
        
        # ============================================================
        # 分类
        # ============================================================
        out = self.classifier(decoded)
        
        return out, min_len