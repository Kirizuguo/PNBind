
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import PointGNNConv, global_mean_pool as gep

# ====================================================================
# 基础模块（复用）
# ====================================================================

class mlp_h(nn.Module):
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
# 🔴 核心创新：轻量级多尺度局部Cross-Attention
# ====================================================================

class LightweightMultiScaleCrossAttn(nn.Module):
    """
    轻量级多尺度局部Cross-Attention
    
    核心设计：
    1. 完全共享 Q/K/V/Out 投影矩阵（参数不增加）
    2. 只有k-NN的k值不同（捕获不同尺度）
    3. 用可学习权重加权平均（只增加2个参数）
    
    理论：
    - k=8:  捕获局部化学环境（侧链相互作用）
    - k=16: 捕获二级结构（α-helix, β-sheet）
    
    参数增加：<10个（几乎为0）
    """
    def __init__(self, geo_dim=1536, sem_dim=512, rsa_dim=64, 
                 output_dim=256, num_heads=8, scales=[8, 16], dropout=0.2):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scales = scales  # [8, 16]
        
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
        
        # 🔴 完全共享的 Q/K/V 投影（关键！）
        # ✅ 改为部分共享
        self.q_proj_geo = nn.Linear(output_dim, output_dim)
        self.q_proj_sem = nn.Linear(output_dim, output_dim)

        # Key 和 Value 依然共享
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # 输出层也共享
        self.out_proj = nn.Linear(output_dim, output_dim)

        
        # 🔴 只增加这2个参数！可学习的尺度权重
        self.scale_weights_geo = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_weights_sem = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
        # Gated Residual
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
        """构建k-NN邻域mask"""
        L = pos.size(0)
        dist = torch.cdist(pos, pos)  # [L, L]
        _, indices = torch.topk(dist, min(k, L), largest=False, dim=1)
        mask = torch.zeros(L, L, dtype=torch.bool, device=pos.device)
        mask.scatter_(1, indices, True)
        return mask
    
    def local_attention(self, query, key, value, mask):
        """局部masked attention"""
        # 转换维度: [L, num_heads, head_dim] -> [num_heads, L, head_dim]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, L, L]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, value)  # [num_heads, L, head_dim]
        
        # 转换回来
        output = output.transpose(0, 1)  # [L, num_heads, head_dim]
        
        return output
    
    def multiscale_cross_attention(self, query_feat, key_feat, pos, scale_weights, query_type='geo'):
        """
        多尺度Cross-Attention
        
        Args:
            query_feat: [L, output_dim] 查询特征（geo或sem）
            key_feat: [L, output_dim] 键值特征（sem或geo）
            pos: [L, 3] 3D坐标
            scale_weights: 可学习的尺度权重
        
        Returns:
            fused: [L, output_dim] 融合后的特征
        """
        L = query_feat.size(0)
        
        # 🔴 共享的Q/K/V投影
        if query_type == 'geo':
            Q = self.q_proj_geo(query_feat).view(L, self.num_heads, self.head_dim)
        else:
            Q = self.q_proj_sem(query_feat).view(L, self.num_heads, self.head_dim)

        K = self.k_proj(key_feat).view(L, self.num_heads, self.head_dim)
        V = self.v_proj(key_feat).view(L, self.num_heads, self.head_dim)

        
        # 🔴 多尺度并行计算
        scale_outputs = []
        for k in self.scales:
            mask = self.build_knn_mask(pos, k)
            out = self.local_attention(Q, K, V, mask)
            out = out.reshape(L, self.output_dim)
            scale_outputs.append(out)
        
        # 🔴 可学习加权平均（而不是concat）
        weights = F.softmax(scale_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, scale_outputs))
        
        # 🔴 共享的输出投影
        fused = self.out_proj(fused)
        
        return fused
    
    def forward(self, geo_feat, sem_feat, rsa_feat, pos):
        """
        🔧 并行双向 Cross-Attention（共享一次 K/V）
        """
        # ===== 统一投影 =====
        geo_proj = self.geo_proj(geo_feat)  # [L, 256]
        sem_proj = self.sem_proj(sem_feat)  # [L, 256]
        rsa_proj = self.rsa_proj(rsa_feat)  # [L, 256]

        L = geo_proj.size(0)

        # ===== 构建共享的 K/V =====
        # 拼接后一次性投影，避免重复计算
        joint_feat = torch.cat([geo_proj, sem_proj], dim=0)  # [2L, 256]
        K = self.k_proj(joint_feat).view(2 * L, self.num_heads, self.head_dim)
        V = self.v_proj(joint_feat).view(2 * L, self.num_heads, self.head_dim)
        K = F.layer_norm(K, K.shape[-1:])
        V = F.layer_norm(V, V.shape[-1:])

        # ===== 生成各自的 Query =====
        Q_geo = self.q_proj_geo(geo_proj).view(L, self.num_heads, self.head_dim)
        Q_sem = self.q_proj_sem(sem_proj).view(L, self.num_heads, self.head_dim)

        # ===== 各尺度并行注意力 =====
        geo_scale_outs, sem_scale_outs = [], []
        for k in self.scales:
            mask = self.build_knn_mask(pos, k)  # [L, L]
            # 扩展到 [L, 2L]，前半（geo）有局部mask，后半（sem）全True
            mask_geo = torch.cat([mask, torch.ones_like(mask)], dim=1)  # [L, 2L]
            mask_sem = torch.cat([torch.ones_like(mask), mask], dim=1)  # [L, 2L]

            geo_scale_outs.append(self.local_attention(Q_geo, K, V, mask_geo))
            sem_scale_outs.append(self.local_attention(Q_sem, K, V, mask_sem))


        # ===== 多尺度加权融合 =====
        w_geo = F.softmax(self.scale_weights_geo, dim=0)
        w_sem = F.softmax(self.scale_weights_sem, dim=0)
        geo_attn_out = sum(w * out.reshape(L, self.output_dim) for w, out in zip(w_geo, geo_scale_outs))
        sem_attn_out = sum(w * out.reshape(L, self.output_dim) for w, out in zip(w_sem, sem_scale_outs))

        # ===== 输出投影 + 残差 =====
        geo_attn_out = self.out_proj(geo_attn_out)
        sem_attn_out = self.out_proj(sem_attn_out)

        geo_output = self.layer_norm1(self.alpha_geo * geo_proj + self.beta_geo * geo_attn_out)
        sem_output = self.layer_norm2(self.alpha_sem * sem_proj + self.beta_sem * sem_attn_out)

        # ===== RSA独立处理 =====
        rsa_output = self.rsa_transform(rsa_proj)

        # ===== 最终融合 =====
        combined = torch.cat([geo_output, sem_output, rsa_output], dim=1)
        fused = self.fusion(combined)

        return fused



# ====================================================================
# T-Net
# ====================================================================

class Tnet(nn.Module):
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
    PNBind - 轻量级多尺度局部Cross-Attention版本
    
    关键改进：
    1. 多尺度局部attention（k=8, k=16）
    2. 完全共享权重（参数几乎不增加）
    3. 可学习的尺度加权
    """
    
    def __init__(self, esm_dim=1536, esmc_dim=1152, num_classes=2, dropout_rate=0.2, scales=[8, 16,24]):
        super().__init__()
        
        self.esm_dim = esm_dim
        self.esmc_dim = esmc_dim
        self.scales = scales
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
        # 🔴 ESMC 增强模块
        # ================================================================
        if esmc_dim > 0:
            # 可学习的放大系数
            self.esmc_scale = nn.Parameter(torch.tensor(2.5))
            
            # ESMC 独立预处理网络
            self.esmc_preprocess = nn.Sequential(
                nn.Linear(esmc_dim, esmc_dim),           # 1152 → 1152
                nn.LayerNorm(esmc_dim),
                nn.Dropout(dropout_rate * 0.5),
                nn.LeakyReLU(),
                nn.Linear(esmc_dim, esmc_dim * 2),       # 1152 → 2304
                nn.LayerNorm(esmc_dim * 2),
                nn.LeakyReLU()
            )
        else:
            self.esmc_scale = None
            self.esmc_preprocess = None
        # ================================================================
        # 语义分支
        # ================================================================
        # 🔴 根据是否使用 ESMC 调整输入维度
        if esmc_dim > 0:
            # ESM3(1536) + ESMC处理后(2304) + HMM(30) + PSSM(20) + DSSP(13) = 3903
            semantic_input_dim = esm_dim + esmc_dim * 2 + 30 + 20 + 13
        else:
            semantic_input_dim = esm_dim + 30 + 20 + 13

        semantic_output_dim = 512
        
        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_input_dim, 1536),      # 🔴 1024 → 1536
            nn.LayerNorm(1536),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(1536, 1024),                    # 🔴 768 → 1024
            nn.LayerNorm(1024),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(1024, semantic_output_dim),
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
        # 🔴 轻量级多尺度局部Cross-Attention融合层
        # ================================================================
        fusion_output_dim = 256
        
        self.fusion_layer = LightweightMultiScaleCrossAttn(
            geo_dim=esm_dim,
            sem_dim=semantic_output_dim,
            rsa_dim=rsa_output_dim,
            output_dim=fusion_output_dim,
            num_heads=8,
            scales=scales,
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_graph):
        # 提取输入
        edge_index = point_graph.edge_index
        pos = point_graph.pos
        x_esm = point_graph.x_esm3
        x_esmc = point_graph.x_esmc if hasattr(point_graph, 'x_esmc') else None
        hmm = point_graph.hmm
        pssm = point_graph.pssm
        secondary = point_graph.secondary
        asa = point_graph.asa
        
        # 长度对齐
        # 长度对齐
        if x_esmc is not None:
            min_len = min(
                pos.size(0), x_esm.size(0), x_esmc.size(0), hmm.size(0),
                pssm.size(0), secondary.size(0), asa.size(0)
            )
        else:
            min_len = min(
                pos.size(0), x_esm.size(0), hmm.size(0),
                pssm.size(0), secondary.size(0), asa.size(0)
            )
        
        pos = pos[:min_len]
        x_esm = x_esm[:min_len]
        if x_esmc is not None:                        # 🔴 新增
            x_esmc = x_esmc[:min_len]                 # 🔴 新增
        hmm = hmm[:min_len]
        pssm = pssm[:min_len]
        secondary = secondary[:min_len]
        asa = asa[:min_len]
        edge_mask = (edge_index[0] < min_len) & (edge_index[1] < min_len)
        edge_index = edge_index[:, edge_mask]
        
        # ============================================================
        # 几何分支
        # ============================================================
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_normalized = pos_centered / (pos_centered.std(dim=0, keepdim=True) + 1e-6)
        
        T = self.pos_tnet(pos_normalized)
        pos_transformed = torch.matmul(pos_normalized, T)
        
        pos_embed = self.pos_proj(F.layer_norm(pos_transformed, (3,)))
        
        geo_feat = pos_embed
        for layer in self.gnn_layers:
            geo_feat = self.LeakyReLU(layer(geo_feat, pos_embed, edge_index))
        
        # ============================================================
        # 语义分支
        # ============================================================

        # 🔴 处理 ESM3 和 ESMC
        if x_esmc is not None and self.esmc_preprocess is not None:
            # ESMC 单独处理：放大 → 预处理
            x_esmc_scaled = x_esmc * self.esmc_scale                    # 放大
            x_esmc_processed = self.esmc_preprocess(x_esmc_scaled)      # 1152 → 2304
            
            # 与 ESM3 拼接
            x_plm = torch.cat([x_esm, x_esmc_processed], dim=-1)       # [N, 1536+2304=3840]
        else:
            x_plm = x_esm                                                # [N, 1536]

        # 拼接所有语义特征
        semantic_input = torch.cat([x_plm, hmm, pssm, secondary], dim=1)

        # 通过语义分支
        semantic_feat = self.semantic_branch(semantic_input)
        
        # ============================================================
        # RSA分支
        # ============================================================
        rsa_feat = self.rsa_branch(asa.unsqueeze(1))
        
        # ============================================================
        # 🔴 多尺度局部Cross-Attention融合
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