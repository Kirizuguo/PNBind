import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import PointGNNConv, global_mean_pool as gep

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


#Cross-Attention

class LightweightMultiScaleCrossAttn(nn.Module):

    def __init__(self, geo_dim=1536, sem_dim=512, rsa_dim=64, 
                 output_dim=256, num_heads=8, scales=[8, 16], dropout=0.2):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scales = scales  # [8, 16]
        
        
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
             
        self.q_proj_geo = nn.Linear(output_dim, output_dim)
        self.q_proj_sem = nn.Linear(output_dim, output_dim)
        
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        self.out_proj = nn.Linear(output_dim, output_dim)

        self.scale_weights_geo = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_weights_sem = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
        # Gated Residual
        self.alpha_geo = nn.Parameter(torch.tensor(1.0))
        self.beta_geo = nn.Parameter(torch.tensor(0.5))
        self.alpha_sem = nn.Parameter(torch.tensor(1.0))
        self.beta_sem = nn.Parameter(torch.tensor(0.5))
        
        # RSA
        self.rsa_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU()
        )
        
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
        L = pos.size(0)
        dist = torch.cdist(pos, pos)  # [L, L]
        _, indices = torch.topk(dist, min(k, L), largest=False, dim=1)
        mask = torch.zeros(L, L, dtype=torch.bool, device=pos.device)
        mask.scatter_(1, indices, True)
        return mask
    
    def local_attention(self, query, key, value, mask):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, L, L]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, value)  # [num_heads, L, head_dim]
        
        output = output.transpose(0, 1)  # [L, num_heads, head_dim]
        
        return output
    
    def multiscale_cross_attention(self, query_feat, key_feat, pos, scale_weights, query_type='geo'):
        L = query_feat.size(0)
        
        if query_type == 'geo':
            Q = self.q_proj_geo(query_feat).view(L, self.num_heads, self.head_dim)
        else:
            Q = self.q_proj_sem(query_feat).view(L, self.num_heads, self.head_dim)

        K = self.k_proj(key_feat).view(L, self.num_heads, self.head_dim)
        V = self.v_proj(key_feat).view(L, self.num_heads, self.head_dim)

        
        scale_outputs = []
        for k in self.scales:
            mask = self.build_knn_mask(pos, k)
            out = self.local_attention(Q, K, V, mask)
            out = out.reshape(L, self.output_dim)
            scale_outputs.append(out)
        
        weights = F.softmax(scale_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, scale_outputs))
        
        fused = self.out_proj(fused)
        
        return fused
    
    def forward(self, geo_feat, sem_feat, rsa_feat, pos):

        geo_proj = self.geo_proj(geo_feat)  # [L, 256]
        sem_proj = self.sem_proj(sem_feat)  # [L, 256]
        rsa_proj = self.rsa_proj(rsa_feat)  # [L, 256]

        L = geo_proj.size(0)

        joint_feat = torch.cat([geo_proj, sem_proj], dim=0)  # [2L, 256]
        K = self.k_proj(joint_feat).view(2 * L, self.num_heads, self.head_dim)
        V = self.v_proj(joint_feat).view(2 * L, self.num_heads, self.head_dim)
        K = F.layer_norm(K, K.shape[-1:])
        V = F.layer_norm(V, V.shape[-1:])

        # Query
        Q_geo = self.q_proj_geo(geo_proj).view(L, self.num_heads, self.head_dim)
        Q_sem = self.q_proj_sem(sem_proj).view(L, self.num_heads, self.head_dim)

        geo_scale_outs, sem_scale_outs = [], []
        for k in self.scales:
            mask = self.build_knn_mask(pos, k)  # [L, L]
            mask_geo = torch.cat([mask, torch.ones_like(mask)], dim=1)  # [L, 2L]
            mask_sem = torch.cat([torch.ones_like(mask), mask], dim=1)  # [L, 2L]

            geo_scale_outs.append(self.local_attention(Q_geo, K, V, mask_geo))
            sem_scale_outs.append(self.local_attention(Q_sem, K, V, mask_sem))


        w_geo = F.softmax(self.scale_weights_geo, dim=0)
        w_sem = F.softmax(self.scale_weights_sem, dim=0)
        geo_attn_out = sum(w * out.reshape(L, self.output_dim) for w, out in zip(w_geo, geo_scale_outs))
        sem_attn_out = sum(w * out.reshape(L, self.output_dim) for w, out in zip(w_sem, sem_scale_outs))

        geo_attn_out = self.out_proj(geo_attn_out)
        sem_attn_out = self.out_proj(sem_attn_out)

        geo_output = self.layer_norm1(self.alpha_geo * geo_proj + self.beta_geo * geo_attn_out)
        sem_output = self.layer_norm2(self.alpha_sem * sem_proj + self.beta_sem * sem_attn_out)

        rsa_output = self.rsa_transform(rsa_proj)

        combined = torch.cat([geo_output, sem_output, rsa_output], dim=1)
        fused = self.fusion(combined)

        return fused

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

def positional_encoding(d_model, max_len=7000):
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros((max_len, d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class benchmark(nn.Module):
   
    
    def __init__(self, esm_dim=1536, esmc_dim=1152, esm2_dim=1280, esm1b_dim=1280, esm2_3b_dim=2560, num_classes=2, dropout_rate=0.2, scales=[8, 16,24]):
        super().__init__() 

        self.esm_dim = esm_dim
        self.esmc_dim = esmc_dim
        self.esm2_dim = esm2_dim
        self.esm1b_dim = esm1b_dim     
        self.esm2_3b_dim = esm2_3b_dim  
        self.scales = scales
        self.dropout = nn.Dropout(dropout_rate)
        self.LeakyReLU = nn.LeakyReLU()
        
        GEO_DIM = 1536  

        self.pos_tnet = Tnet(input_dim=3)
        self.pos_proj = nn.Linear(3, GEO_DIM)

        self.gnn_layers = nn.ModuleList([
            PointGNNConv(
                mlp_h(GEO_DIM, GEO_DIM//2, GEO_DIM, dropout_rate),
                mlp_f(GEO_DIM*2, GEO_DIM, GEO_DIM, dropout_rate),
                mlp_g(GEO_DIM, GEO_DIM, GEO_DIM, dropout_rate)
            )
            for _ in range(3)
        ])

        if esmc_dim > 0:
            self.esmc_scale = nn.Parameter(torch.tensor(2.5))

            self.esmc_preprocess = nn.Sequential(
                nn.Linear(esmc_dim, esmc_dim),           
                nn.LayerNorm(esmc_dim),
                nn.Dropout(dropout_rate * 0.5),
                nn.LeakyReLU(),
                nn.Linear(esmc_dim, esmc_dim * 2),      
                nn.LayerNorm(esmc_dim * 2),
                nn.LeakyReLU()
            )
        else:
            self.esmc_scale = None
            self.esmc_preprocess = None

        if esm2_dim > 0:
            self.esm2_scale = nn.Parameter(torch.tensor(2.0))
            self.esm2_preprocess = nn.Sequential(
                nn.Linear(esm2_dim, esm2_dim),
                nn.LayerNorm(esm2_dim),
                nn.Dropout(dropout_rate * 0.5),
                nn.LeakyReLU(),
                nn.Linear(esm2_dim, esm2_dim * 2),  # 1280 → 2560
                nn.LayerNorm(esm2_dim * 2),
                nn.LeakyReLU()
            )
        else:
            self.esm2_scale = None
            self.esm2_preprocess = None

        if esm1b_dim > 0:
            self.esm1b_scale = nn.Parameter(torch.tensor(2.0))
            self.esm1b_preprocess = nn.Sequential(
                nn.Linear(esm1b_dim, esm1b_dim),
                nn.LayerNorm(esm1b_dim),
                nn.Dropout(dropout_rate * 0.5),
                nn.LeakyReLU(),
                nn.Linear(esm1b_dim, esm1b_dim * 2),  # 1280 → 2560
                nn.LayerNorm(esm1b_dim * 2),
                nn.LeakyReLU()
            )
        else:
            self.esm1b_scale = None
            self.esm1b_preprocess = None
            

        if esm2_3b_dim > 0:
            self.esm2_3b_scale = nn.Parameter(torch.tensor(3.0))  
            self.esm2_3b_preprocess = nn.Sequential(
                nn.Linear(esm2_3b_dim, esm2_3b_dim),
                nn.LayerNorm(esm2_3b_dim),
                nn.Dropout(dropout_rate * 0.5),
                nn.LeakyReLU(),
                nn.Linear(esm2_3b_dim, esm2_3b_dim * 2),  
                nn.LayerNorm(esm2_3b_dim * 2),
                nn.LeakyReLU()
            )
        else:
            self.esm2_3b_scale = None
            self.esm2_3b_preprocess = None
       
        semantic_input_dim = 30 + 20 + 13  

        # ESM3
        if esm_dim > 0:
            semantic_input_dim += esm_dim  

        # ESMC
        if esmc_dim > 0:
            semantic_input_dim += esmc_dim * 2  

        # ESM-2
        if esm2_dim > 0:
            semantic_input_dim += esm2_dim * 2  
        
        # ESM1b
        if esm1b_dim > 0:
            semantic_input_dim += esm1b_dim * 2  
            
        # ESM2-3B
        if esm2_3b_dim > 0:
            semantic_input_dim += esm2_3b_dim * 2  

        print(f"[Model] Semantic input dim: {semantic_input_dim}")

        semantic_output_dim = 512

        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_input_dim, 2048),  
            nn.LayerNorm(2048),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(1024, semantic_output_dim),
            nn.LayerNorm(semantic_output_dim),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU()
        )
        
        rsa_output_dim = 64
        self.rsa_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, rsa_output_dim),
            nn.LayerNorm(rsa_output_dim),
            nn.LeakyReLU()
        )
        

        fusion_output_dim = 256
        
        self.fusion_layer = LightweightMultiScaleCrossAttn(
            geo_dim=1536,
            sem_dim=semantic_output_dim,
            rsa_dim=rsa_output_dim,
            output_dim=fusion_output_dim,
            num_heads=8,
            scales=scales,
            dropout=dropout_rate
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_output_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout_rate,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

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
                if m.weight.shape[0] > 0 and m.weight.shape[1] > 0:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, point_graph):

        edge_index = point_graph.edge_index
        pos = point_graph.pos
        x_esm = point_graph.x_esm3
        x_esmc = point_graph.x_esmc if hasattr(point_graph, 'x_esmc') else None
        x_esm2 = point_graph.x_esm2 if hasattr(point_graph, 'x_esm2') else None
        x_esm1b = point_graph.x_esm1b if hasattr(point_graph, 'x_esm1b') else None      # 新增
        x_esm2_3b = point_graph.x_esm2_3b if hasattr(point_graph, 'x_esm2_3b') else None  # 新增
        hmm = point_graph.hmm
        pssm = point_graph.pssm
        secondary = point_graph.secondary
        asa = point_graph.asa

        min_len = pos.size(0)
        for feat in [x_esm, x_esmc, x_esm2, x_esm1b, x_esm2_3b, hmm, pssm, secondary, asa]:
            if feat is not None:
                min_len = min(min_len, feat.size(0))

        pos = pos[:min_len]
        x_esm = x_esm[:min_len]
        if x_esmc is not None:
            x_esmc = x_esmc[:min_len]
        if x_esm2 is not None:  
            x_esm2 = x_esm2[:min_len]
        if x_esm1b is not None:    
            x_esm1b = x_esm1b[:min_len]
        if x_esm2_3b is not None: 
            x_esm2_3b = x_esm2_3b[:min_len]
        hmm = hmm[:min_len]
        pssm = pssm[:min_len]
        secondary = secondary[:min_len]
        asa = asa[:min_len]
        edge_mask = (edge_index[0] < min_len) & (edge_index[1] < min_len)
        edge_index = edge_index[:, edge_mask]
                
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        pos_normalized = pos_centered / (pos_centered.std(dim=0, keepdim=True) + 1e-6)
        
        T = self.pos_tnet(pos_normalized)
        pos_transformed = torch.matmul(pos_normalized, T)
        
        pos_embed = self.pos_proj(F.layer_norm(pos_transformed, (3,)))
        
        geo_feat = pos_embed
        for layer in self.gnn_layers:
            geo_feat = self.LeakyReLU(layer(geo_feat, pos_embed, edge_index))
        
        plm_features = []

        # ESM3 
        if self.esm_dim > 0 and hasattr(point_graph, "x_esm3"):
            plm_features.append(x_esm)

                
        # ESMC
        if x_esmc is not None and self.esmc_preprocess is not None:
            x_esmc_scaled = x_esmc * self.esmc_scale
            x_esmc_processed = self.esmc_preprocess(x_esmc_scaled)  # [N, 2304]
            plm_features.append(x_esmc_processed)
        
        # ESM-2
        if x_esm2 is not None and self.esm2_preprocess is not None:
            x_esm2_scaled = x_esm2 * self.esm2_scale
            x_esm2_processed = self.esm2_preprocess(x_esm2_scaled)  # [N, 2560]
            plm_features.append(x_esm2_processed)
        # ESM1b
        if x_esm1b is not None and self.esm1b_preprocess is not None:
            x_esm1b_scaled = x_esm1b * self.esm1b_scale
            x_esm1b_processed = self.esm1b_preprocess(x_esm1b_scaled)  # [N, 2560]
            plm_features.append(x_esm1b_processed)
        
        # ESM2-3B
        if x_esm2_3b is not None and self.esm2_3b_preprocess is not None:
            x_esm2_3b_scaled = x_esm2_3b * self.esm2_3b_scale
            x_esm2_3b_processed = self.esm2_3b_preprocess(x_esm2_3b_scaled)  # [N, 5120]
            plm_features.append(x_esm2_3b_processed)

        x_plm = torch.cat(plm_features, dim=-1)  

        semantic_input = torch.cat([x_plm, hmm, pssm, secondary], dim=1)
        semantic_feat = self.semantic_branch(semantic_input)         
        rsa_feat = self.rsa_branch(asa.unsqueeze(1))
        fused = self.fusion_layer(geo_feat, semantic_feat, rsa_feat, pos_normalized)

        pe = positional_encoding(fused.size(1), min_len).to(fused.device)
        fused_with_pe = fused + pe[:min_len, :fused.size(1)]
        
        fused_t = fused_with_pe.unsqueeze(1)
        decoded = self.decoder(fused_t, fused_t).squeeze(1)
        
        out = self.classifier(decoded)
        
        return out, min_len