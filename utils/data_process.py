# data_process.py
_esm3_model_cache = None
_esmc_model_cache = None
_esm2_model_cache = None      # 新增
_esm2_alphabet_cache = None   # 新增
_esm1b_model_cache = None      # 新增
_esm1b_alphabet_cache = None   # 新增
_esm2_3b_model_cache = None    # 新增
_esm2_3b_alphabet_cache = None # 新增
import sys, os
import matplotlib.pyplot as plt

def analyze_sequence_lengths():
    """统计数据集中的序列长度分布"""
    lengths = []
    
    for dataset, split in [("prna", "train"), ("prna", "test"), ("pdna", "train"), ("pdna", "test")]:
        label_dir = f"{dataset}_labels/{dataset}_{split}_label_onlyc"
        if not os.path.exists(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            with open(os.path.join(label_dir, file), 'r') as f:
                lines = f.readlines()
                lengths.append(len(lines))
    
    lengths = sorted(lengths)
    print(f"\n{'='*60}")
    print(f"序列长度统计 (总计{len(lengths)}个样本)")
    print(f"{'='*60}")
    print(f"最短: {min(lengths)}")
    print(f"最长: {max(lengths)}")
    print(f"平均: {sum(lengths)/len(lengths):.1f}")
    print(f"中位数: {lengths[len(lengths)//2]}")
    print(f"\n超过1024的样本数: {sum(1 for l in lengths if l > 1024)} ({sum(1 for l in lengths if l > 1024)/len(lengths)*100:.1f}%)")
    print(f"超过2048的样本数: {sum(1 for l in lengths if l > 2048)} ({sum(1 for l in lengths if l > 2048)/len(lengths)*100:.1f}%)")
    
    return lengths

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("[PATH FIX] Added:", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
from torch_cluster import radius_graph
import json
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer, pipeline, AlbertModel, AlbertTokenizer, XLNetTokenizer, XLNetModel
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser
import esm
import sys
sys.path.insert(0, "/home/ghd/PNBind/GAPointnet_pytorch/esm3")
from esm3.models.esm3 import ESM3
from esm3.sdk.api import ESMProtein,  LogitsConfig
from esm3.models.esmc import ESMC


res_dict = {
    'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'TRP': 'W',
    'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
    'LYS': 'K', 'ARG': 'R', 'Unknown': 'X'
}
# ======== 新增：非法残基修复映射 ========
AA_FIX_MAP = {
    "A": "A", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H",
    "I": "I", "K": "K", "L": "L", "M": "M", "N": "N", "P": "P", "Q": "Q",
    "R": "R", "S": "S", "T": "T", "V": "V", "W": "W", "Y": "Y",
    "U": "S", "T": "T", "G": "G", "C": "C", "A": "A",  # DNA/RNA碱基 → 类似氨基酸
    "X": "G", "N": "G", "-": "G", "*": "G", ".": "G", "?": "G"
}

def fix_sequence(seq_list):
    """将DNA碱基或非法符号映射为合法氨基酸，避免ESM报错"""
    return [AA_FIX_MAP.get(a.upper(), "A") for a in seq_list]

pro_res_table = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
]

atom_type_onehot = {'N': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'O': [0, 0, 1, 0, 0], 'S': [0, 0, 0, 1, 0], 'H': [0, 0, 0, 0, 1]}


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = 'X'
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# prottrans
def prot_xlnet_pretrain(seq, device):
    seq = ' '.join(seq)
    model_name = "Rostlab/prot_xlnet"
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = AlbertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = XLNetModel.from_pretrained(model_name)
    model_protein = model_protein.to(device)
    model_protein = model_protein.eval()
    fe = pipeline('feature-extraction', model=model_protein, tokenizer=tokenizer, device=device)
    embedding = fe(seq)
    embedding = np.array(embedding)
    embedding = embedding.reshape(embedding.shape[1], embedding.shape[2])
    seq_len = len(seq.replace(" ", ""))
    if "t5" in model_name:
        start_Idx = 0
        end_Idx = seq_len
        pc_feature = embedding[start_Idx:end_Idx]
    elif "albert" in model_name:
        start_Idx = 1
        end_Idx = seq_len + 1
        pc_feature = embedding[start_Idx:end_Idx]
    elif "xlnet" in model_name:
        padded_seq_len = len(embedding)
        start_Idx = padded_seq_len - seq_len - 2
        end_Idx = padded_seq_len - 2
        pc_feature = embedding[start_Idx:end_Idx]
    return pc_feature


def prot_t5_xl_bfd_pretrain(seq, device):
    seq = ' '.join(seq)
    model_name = "Rostlab/prot_t5_xl_bfd"
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = AlbertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = XLNetModel.from_pretrained(model_name)
    model_protein = model_protein.to(device)
    model_protein = model_protein.eval()
    fe = pipeline('feature-extraction', model=model_protein, tokenizer=tokenizer, device=device)
    embedding = fe(seq)
    embedding = np.array(embedding)
    embedding = embedding.reshape(embedding.shape[1], embedding.shape[2])
    seq_len = len(seq.replace(" ", ""))
    if "t5" in model_name:
        start_Idx = 0
        end_Idx = seq_len
        pc_feature = embedding[start_Idx:end_Idx]
    elif "albert" in model_name:
        start_Idx = 1
        end_Idx = seq_len + 1
        pc_feature = embedding[start_Idx:end_Idx]
    elif "xlnet" in model_name:
        padded_seq_len = len(embedding)
        start_Idx = padded_seq_len - seq_len - 2
        end_Idx = padded_seq_len - 2
        pc_feature = embedding[start_Idx:end_Idx]
    return pc_feature


def seq_onehot(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
    return pro_hot


# def get_dssp(pdb_path, len_seq):
    p = PDBParser()
    structure = p.get_structure("protein_name", pdb_path)
    model = structure[0]
    dssp = DSSP(model, pdb_path)

    secondary = []
    asa = []
    for res in dssp:
        asa.append(res[3])
        if res[2] in ('G', 'H', 'I'):
            secondary.append([1, 0, 0])
        if res[2] in ('E', 'B'):
            secondary.append([0, 1, 0])
        if res[2] in ('T', 'S', '-'):
            secondary.append([0, 0, 1])

    if len(secondary) != len_seq:
        pdb_path = pdb_path.replace('pdna_pdb', 'pdna_af2_pdb')
        p = PDBParser()
        structure = p.get_structure("protein_name", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path)
        dssp_table = ['G', 'H', 'I', 'E', 'B', 'T', 'S', '-']
        secondary = []
        asa = []
        for res in dssp:
            asa.append(res[3])
            if res[2] in ('G', 'H', 'I'):
                secondary.append([1, 0, 0])
            if res[2] in ('E', 'B'):
                secondary.append([0, 1, 0])
            if res[2] in ('T', 'S', '-'):
                secondary.append([0, 0, 1])
    return secondary, asa

def cal_DSSP_prna(dssp_path, seq):
    """
    ✅ 从 DSSP 文件提取 ASA 与二级结构
    ✅ 兼容固定列宽和空格分隔格式
    ✅ 生成13维secondary特征
    """
    maxASA = {
        'A': 106, 'R': 248, 'N': 157, 'D': 163, 'C': 135, 'E': 194, 'Q': 198,
        'G': 84,  'H': 184, 'I': 169, 'L': 164, 'K': 205, 'M': 188, 'F': 197,
        'P': 136, 'S': 130, 'T': 142, 'W': 227, 'Y': 222, 'V': 142
    }
    
    # 🔴 DSSP 8种二级结构类型映射
    dssp_8_map = {
        ' ': 0, 'S': 1, 'T': 2, 'H': 3, 'G': 4, 
        'I': 5, 'E': 6, 'B': 7, '-': 8
    }
    
    asa_list, s2_list = [], []
    L = len(seq)

    try:
        with open(dssp_path, 'r') as f:
            lines = f.readlines()

        # 🔎 更健壮的头部定位
        start_idx = next(
            (i+1 for i, line in enumerate(lines)
             if "RESIDUE" in line and "STRUCTURE" in line),
            None
        )
        if start_idx is None:
            raise ValueError("Invalid DSSP header")

        for line in lines[start_idx:]:
            if len(line.strip()) == 0:
                continue

            aa, ss, asa_val = None, None, None

            # ① 固定列宽解析
            if len(line) > 38:
                try:
                    aa = line[13].strip()
                    ss = line[16].strip() if len(line) > 16 else ' '
                    asa_val = float(line[34:40].strip())
                except (ValueError, IndexError):
                    pass

            # ② split() 降级解析
            if aa is None or asa_val is None:
                try:
                    parts = line.split()
                    if len(parts) >= 7:
                        aa = parts[2]
                        ss = parts[3] if len(parts) > 3 else ' '
                        asa_val = float(parts[6])
                except (ValueError, IndexError):
                    continue

            if not aa or aa not in maxASA:
                continue

            # ASA归一化
            asa_norm = min(1.0, asa_val / maxASA.get(aa, 200))
            asa_list.append(asa_norm)

            # 🔴 生成13维secondary特征
            ss_idx = dssp_8_map.get(ss, 0)
            ss_onehot = [0.0] * 8
            ss_onehot[ss_idx] = 1.0
            
            # 添加5个占位维度（对齐13维）
            feature = ss_onehot + [0.0] * 5
            s2_list.append(feature)

    except Exception as e:
        print(f"[WARN] Failed to parse {dssp_path}: {e}")
        asa_list = [0.5] * L
        s2_list = [[0.0] * 13] * L  # 🔴 13维默认值

    # ✅ 长度对齐
    if len(asa_list) < L:
        mean_asa = np.mean(asa_list) if asa_list else 0.5
        asa_list += [mean_asa] * (L - len(asa_list))
        s2_list += [[0.0] * 13] * (L - len(s2_list))  # 🔴 13维
    elif len(asa_list) > L:
        asa_list = asa_list[:L]
        s2_list = s2_list[:L]

    return np.array(asa_list), np.array(s2_list)


def call_HMM(hmm_dir): 
    with open(hmm_dir, 'r') as f:
        text = f.readlines()
    hmm_begin_line = 0
    hmm_end_line = 0
    for i in range(len(text)):
        if '#' in text[i]:
            hmm_begin_line = i + 5
        elif '//' in text[i]:
            hmm_end_line = i
    hmm = np.zeros([int((hmm_end_line - hmm_begin_line) / 3), 30])

    axis_x = 0
    for i in range(hmm_begin_line, hmm_end_line, 3):
        line1 = text[i].split()[2:-1]
        line2 = text[i + 1].split()
        axis_y = 0
        for j in line1:
            if j == '*':
                hmm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hmm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        for j in line2:
            if j == '*':
                hmm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hmm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        axis_x += 1
    hmm = (hmm - np.min(hmm)) / (np.max(hmm) - np.min(hmm))

    return hmm


def process_pssm(pssm_file):
    with open(pssm_file, "r") as f:
        lines = f.readlines()
    pssm_feature = []
    for line in lines:
        if line == "\n":
            continue
        record = line.strip().split()
        if record[0].isdigit():
            pssm_feature.append([int(x) for x in record[2:22]])
    pssm_feature = np.array(pssm_feature)
    pssm_feature = (pssm_feature - np.min(pssm_feature)) / (np.max(pssm_feature) - np.min(pssm_feature))
    return pssm_feature




def rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class prna(InMemoryDataset):
    """
    PRNA 数据集构建类 —— 使用 ESM3 提取每残基 embedding
    """
    def __init__(self, root='prna_labels', dataset='prna', data_split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.data_split = data_split
        super(prna, self).__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.processed_paths[0]):
            print(f"[prna] 未检测到缓存，开始处理数据集：{self.dataset}_{self.data_split}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"[prna] 已加载完毕，共 {len(self)} 个样本 ({self.dataset}_{self.data_split})")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.dataset}_{self.data_split}.pt"]

    def get_neibors(self, coors, threshold=8, max_num_neighbors=32):
        return radius_graph(coors, r=threshold, max_num_neighbors=max_num_neighbors)

    def process(self):
        dssp_path = os.path.join(self.root, f"{self.dataset}_dssp")
        hmm_path  = os.path.join(self.root, f"{self.dataset}_hmm")
        pssm_path = os.path.join(self.root, f"{self.dataset}_pssm")

        all_files = os.listdir(os.path.join(self.root, f"{self.dataset}_{self.data_split}_label"))
        graph_datas = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[prna] 使用设备: {device}")

        for one_file in tqdm(all_files, desc=f"[prna] Processing {self.data_split}"):
            if one_file.startswith("4ne1_p"):
                print(f"[prna skip] 跳过缺失样本 {one_file}")
                continue

            # === 载入坐标与标签 ===
            with open(f"{self.root}/{self.dataset}_{self.data_split}_label_onlyc/{one_file}", 'r') as temp:
                data = [line.split() for line in temp]

            coors = [list(map(float, d[0:-2])) for d in data]
            x_features = [d[-2] for d in data]
            y = [int(d[-1]) for d in data]

            # === 修复非法残基 ===
            x_features = fix_sequence(x_features)

            # === 文件路径检查 ===
            dssp_file = os.path.join(dssp_path, one_file.split('.')[0] + '.dssp')
            hmm_file  = os.path.join(hmm_path,  one_file.split('.')[0] + '.hmm')
            pssm_file = os.path.join(pssm_path, one_file.split('.')[0] + '.pssm')

            if not (os.path.exists(dssp_file) and os.path.exists(hmm_file) and os.path.exists(pssm_file)):
                print(f"[prna skip] {one_file}: 缺少 dssp/hmm/pssm 文件")
                continue

            # === 加载模态特征 ===
            asa, s2 = cal_DSSP_prna(dssp_file, x_features)


            hmm     = call_HMM(hmm_file)
            pssm    = process_pssm(pssm_file)
            x_onehot = seq_onehot(x_features)
            
            # === ESM3 embedding（先生成，后对齐）===
            try:
                x_esm3 = esm3_sm_open(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM3 失败 {one_file}: {e}")
                x_esm3 = torch.zeros((len(x_features), 1536))
            # === ESMC embedding ===
            try:
                x_esmc = esmc_600m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESMC 失败 {one_file}: {e}")
                x_esmc = torch.zeros((len(x_features), 1152))
            
            # === ESM-2 embedding (新增) ===
            try:
                x_esm2 = esm2_t33_650m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM2 失败 {one_file}: {e}")
                x_esm2 = torch.zeros((len(x_features), 1280))
            # === ESM1b embedding (新增) ===
            try:
                x_esm1b = esm1b_t33_650m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM1b 失败 {one_file}: {e}")
                x_esm1b = torch.zeros((len(x_features), 1280))

            # === ESM2-3B embedding (新增) ===
            try:
                x_esm2_3b = esm2_t36_3b(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM2-3B 失败 {one_file}: {e}")
                x_esm2_3b = torch.zeros((len(x_features), 2560))
            # === 对齐各模态长度（包括ESM3）===
            min_len = min(len(x_onehot), len(hmm), len(pssm), len(s2), len(asa), len(y), len(x_esm3), len(x_esmc),len(x_esm2), len(x_esm1b), len(x_esm2_3b))
            if min_len == 0:
                print(f"[skip] {one_file}: 空特征，跳过")
                continue

            # 🔴 统一截断所有特征
            x_onehot = x_onehot[:min_len]
            hmm = hmm[:min_len]
            pssm = pssm[:min_len]
            s2 = s2[:min_len]
            asa = asa[:min_len]
            y = y[:min_len]
            x_esm3 = x_esm3[:min_len]
            x_esmc = x_esmc[:min_len]
            x_esm2 = x_esm2[:min_len] 
            x_esm1b = x_esm1b[:min_len]    # 新增
            x_esm2_3b = x_esm2_3b[:min_len] # 新增  # 🔴 关键修复！
            # === 坐标长度对齐 ===
            len_y = len(y)
            if len(coors) != len_y:
                if len(coors) > len_y:
                    coors = coors[:len_y]
                else:
                    coors.extend([[0.0, 0.0, 0.0]] * (len_y - len(coors)))

            coors_tensor = torch.tensor(coors, dtype=torch.float)
            edge_index = self.get_neibors(coors_tensor)

            point_graph = Data(
                x=torch.tensor(x_onehot, dtype=torch.float),
                edge_index=edge_index,
                y=torch.LongTensor(y),
                hmm=torch.tensor(hmm, dtype=torch.float),
                secondary=torch.tensor(s2, dtype=torch.float),
                pssm=torch.tensor(pssm, dtype=torch.float),
                asa=torch.tensor(asa, dtype=torch.float),
                x_esm3=x_esm3,
                x_esmc=x_esmc,
                x_esm2=x_esm2,
                x_esm1b=x_esm1b,      # 新增
                x_esm2_3b=x_esm2_3b,
                pos=coors_tensor,
                protein_name=one_file
            )
            graph_datas.append(point_graph)

        # === 保存数据 ===
        # === 保存数据前统一设备并释放显存 ===
        graph_datas = [g.to('cpu') for g in graph_datas]
        self.data, self.slices = self.collate(graph_datas)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        torch.save((self.data, self.slices), self.processed_paths[0])
        print(f"[{self.dataset}] ✅ 保存完成，共 {len(graph_datas)} 个样本")

        print(f"[prna] 处理完成，共 {len(graph_datas)} 个样本保存到 {self.processed_paths[0]}")

class pdna(InMemoryDataset):
    """
    PDNA 数据集构建类 —— 使用 ESM3 提取每残基 embedding
    """
    def __init__(self, root='pdna_labels', dataset='pdna', data_split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.data_split = data_split
        super(pdna, self).__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.processed_paths[0]):
            print(f"[pdna] 未检测到缓存，开始处理数据集：{self.dataset}_{self.data_split}")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"[pdna] 已加载完毕，共 {len(self)} 个样本 ({self.dataset}_{self.data_split})")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.dataset}_{self.data_split}.pt"]

    def get_neibors(self, coors, threshold=8, max_num_neighbors=32):
        return radius_graph(coors, r=threshold, max_num_neighbors=max_num_neighbors)

    def process(self):
        dssp_path = os.path.join(self.root, f"{self.dataset}_dssp")
        hmm_path  = os.path.join(self.root, f"{self.dataset}_hmm")
        pssm_path = os.path.join(self.root, f"{self.dataset}_pssm")

        all_files = os.listdir(os.path.join(self.root, f"{self.dataset}_{self.data_split}_label"))
        graph_datas = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[pdna] 使用设备: {device}")

        for one_file in tqdm(all_files, desc=f"[pdna] Processing {self.data_split}"):
            if one_file.startswith("4ne1_p"):
                print(f"[pdna skip] 跳过缺失样本 {one_file}")
                continue

            with open(f"{self.root}/{self.dataset}_{self.data_split}_label_onlyc/{one_file}", 'r') as temp:
                data = [line.split() for line in temp]

            coors = [list(map(float, d[0:-2])) for d in data]
            x_features = [d[-2] for d in data]
            y = [int(d[-1]) for d in data]

            x_features = fix_sequence(x_features)

            dssp_file = os.path.join(dssp_path, one_file.split('.')[0] + '.dssp')
            hmm_file  = os.path.join(hmm_path,  one_file.split('.')[0] + '.hmm')
            pssm_file = os.path.join(pssm_path, one_file.split('.')[0] + '.pssm')

            if not (os.path.exists(dssp_file) and os.path.exists(hmm_file) and os.path.exists(pssm_file)):
                print(f"[pdna skip] {one_file}: 缺少 dssp/hmm/pssm 文件")
                continue

            asa, s2 = cal_DSSP_prna(dssp_file, x_features)


            hmm     = call_HMM(hmm_file)
            pssm    = process_pssm(pssm_file)
            x_onehot = seq_onehot(x_features)
            # === ESM3 embedding（先生成，后对齐）===
            try:
                x_esm3 = esm3_sm_open(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM3 失败 {one_file}: {e}")
                x_esm3 = torch.zeros((len(x_features), 1536))
            # === ESMC embedding ===
            try:
                x_esmc = esmc_600m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESMC 失败 {one_file}: {e}")
                x_esmc = torch.zeros((len(x_features), 1152))
            # === ESM-2 embedding (新增) ===
            try:
                x_esm2 = esm2_t33_650m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM2 失败 {one_file}: {e}")
                x_esm2 = torch.zeros((len(x_features), 1280))
            # === ESM1b embedding (新增) ===
            try:
                x_esm1b = esm1b_t33_650m(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM1b 失败 {one_file}: {e}")
                x_esm1b = torch.zeros((len(x_features), 1280))

            # === ESM2-3B embedding (新增) ===
            try:
                x_esm2_3b = esm2_t36_3b(x_features, device=device)
            except Exception as e:
                print(f"[warn] ESM2-3B 失败 {one_file}: {e}")
                x_esm2_3b = torch.zeros((len(x_features), 2560))
            # === 对齐各模态长度（包括ESM3）===
            min_len = min(len(x_onehot), len(hmm), len(pssm), len(s2), len(asa), len(y), len(x_esm3),len(x_esmc), len(x_esm2), len(x_esm1b), len(x_esm2_3b))
            if min_len == 0:
                print(f"[skip] {one_file}: 空特征，跳过")
                continue

            # 🔴 统一截断所有特征
            x_onehot = x_onehot[:min_len]
            hmm = hmm[:min_len]
            pssm = pssm[:min_len]
            s2 = s2[:min_len]
            asa = asa[:min_len]
            y = y[:min_len]
            x_esm3 = x_esm3[:min_len]  # 🔴 关键修复！
            x_esmc = x_esmc[:min_len]
            x_esm2 = x_esm2[:min_len]
            x_esm1b = x_esm1b[:min_len]    # 新增
            x_esm2_3b = x_esm2_3b[:min_len] # 新增
            len_y = len(y)
            if len(coors) != len_y:
                if len(coors) > len_y:
                    coors = coors[:len_y]
                else:
                    coors.extend([[0.0, 0.0, 0.0]] * (len_y - len(coors)))

            coors_tensor = torch.tensor(coors, dtype=torch.float)
            edge_index = self.get_neibors(coors_tensor)

            point_graph = Data(
                x=torch.tensor(x_onehot, dtype=torch.float),
                edge_index=edge_index,
                y=torch.LongTensor(y),
                hmm=torch.tensor(hmm, dtype=torch.float),
                secondary=torch.tensor(s2, dtype=torch.float),
                pssm=torch.tensor(pssm, dtype=torch.float),
                asa=torch.tensor(asa, dtype=torch.float),
                x_esm3=x_esm3,
                x_esmc=x_esmc,
                x_esm2=x_esm2,
                x_esm1b=x_esm1b,      # 新增
                x_esm2_3b=x_esm2_3b,  # 新增
                pos=coors_tensor,
                protein_name=one_file
            )
            graph_datas.append(point_graph)

        self.data, self.slices = self.collate(graph_datas)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print(f"[pdna] 处理完成，共 {len(graph_datas)} 个样本保存到 {self.processed_paths[0]}")

def esm3_sm_open(seq, device='cpu'):
    """
    从 ESM3-sm-open-v1 提取每个氨基酸的 1536维 embedding。
    ✅ 修复：不再触发结构感知模式（避免每原子 embedding）
    ✅ 增强：全局 z-score 归一化 + clamp(-5,5) 限幅
    ✅ 输出 shape = [L, 1536], 已标准化且无显存泄漏
    """
    import torch
    import torch.nn.functional as F
    import gc
    from esm3.models.esm3 import ESM3
    from esm3.sdk.api import get_esm3_model_tokenizers

    global _esm3_model_cache
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)

    legal_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_fixed = ['A' if aa.upper() not in legal_aa else aa.upper() for aa in seq]
    seq_str = ''.join(seq_fixed)

    try:
        # 1️⃣ 模型缓存与加载
        if _esm3_model_cache is None:
            print(f"[ESM3] Loading esm3_sm_open_v1 model ...")
            model = ESM3.from_pretrained("/home/ghd/PNBind/GAPointnet_pytorch/esm3/model")
            model.tokenizers = get_esm3_model_tokenizers()
            model = model.float()
            _esm3_model_cache = model
            print(f"[ESM3] ✅ Model cached successfully")

        model = _esm3_model_cache.to(device)
        model.eval()

        # 2️⃣ 编码与推理
        tokens = model.tokenizers.sequence.encode(seq_str, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens], device=device)

        with torch.no_grad():
            output = model(sequence_tokens=tokens_tensor)

        # 3️⃣ 提取 embedding（去掉BOS/EOS）
        emb = output.embeddings[0, 1:-1, :]
        emb = F.layer_norm(emb, emb.shape[1:])  # 局部归一化（每残基内部）

        # 4️⃣ 全局标准化 + clamp
        mean = emb.mean()
        std = emb.std()
        emb = (emb - mean) / (std + 1e-6)
        emb = torch.clamp(emb, -5, 5)

        # 5️⃣ 显存清理
        emb = emb.detach().cpu()
        del output, tokens_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return emb

    except Exception as e:
        print(f"[ESM3] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return torch.zeros((len(seq_fixed), 1536))

def esmc_600m(seq, device='cpu'):
    """
    从 ESMC-600M 提取每个氨基酸的 embedding (1152维)
    ✅ 从本地权重文件加载
    ✅ 全局模型缓存
    ✅ 归一化处理
    """
    import torch
    import torch.nn.functional as F
    import gc

    global _esmc_model_cache
    
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)

    legal_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_fixed = ['A' if aa.upper() not in legal_aa else aa.upper() for aa in seq]
    seq_str = ''.join(seq_fixed)

    try:
        # 1️⃣ 模型缓存与加载
        if _esmc_model_cache is None:
            print(f"[ESMC] Loading esmc_600m model from local weights...")
            
            # 🔴 从你的本地权重文件加载
            model_path = "/home/ghd/PNBind/GAPointnet_pytorch/esm3/data/weights/esmc_600m_2024_12_v0.pth"
            
            # 加载模型
            model = ESMC.from_pretrained("esmc_600m")  # 先加载结构
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.float()
            
            _esmc_model_cache = model
            print(f"[ESMC] ✅ Model cached successfully from {model_path}")

        model = _esmc_model_cache.to(device)
        model.eval()

        # 2️⃣ 编码与推理
        protein = ESMProtein(sequence=seq_str)
        protein_tensor = model.encode(protein)
        
        with torch.no_grad():
            logits_output = model.logits(
                protein_tensor, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )

        # 3️⃣ 提取 embedding（去掉BOS/EOS）
        emb = logits_output.embeddings.squeeze(0)
        if len(emb) > len(seq_str):
            emb = emb[1:len(seq_str)+1]  # 去掉特殊token
        else:
            emb = emb[:len(seq_str)]
        
        # 4️⃣ 归一化处理（与ESM3保持一致）
        emb = F.layer_norm(emb, emb.shape[1:])
        mean = emb.mean()
        std = emb.std()
        emb = (emb - mean) / (std + 1e-6)
        emb = torch.clamp(emb, -10, 10)

        # 5️⃣ 显存清理
        emb = emb.detach().cpu()
        del logits_output, protein_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return emb

    except Exception as e:
        print(f"[ESMC] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return torch.zeros((len(seq_fixed), 1152))  # ESMC是1152维

def esm2_t33_650m(seq, device='cpu'):
    """
    从本地权重加载 ESM-2 t33 650M，提取1280维embedding
    """
    import torch
    import torch.nn.functional as F
    import gc
    import esm
    from esm.pretrained import load_model_and_alphabet_core
    
    global _esm2_model_cache, _esm2_alphabet_cache
    
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)
    
    legal_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_fixed = ['A' if aa.upper() not in legal_aa else aa.upper() for aa in seq]
        # 🔴 新增：截断超长序列
    original_len = len(seq_fixed)
    MAX_LEN = 1022
    if original_len > MAX_LEN:
        seq_fixed = seq_fixed[:MAX_LEN]
        print(f"[ESM2] Truncated {original_len} -> {MAX_LEN}")
    seq_str = ''.join(seq_fixed)
    
    try:
        # 1️⃣ 模型缓存与加载
        if _esm2_model_cache is None:
            print(f"[ESM2] Loading esm2_t33_650M from local weights...")
            model_path = "/home/ghd/PNBind/GAPointnet_pytorch/esm3/data/weights/esm2_t33_650M_UR50D.pt"
            model_data = torch.load(model_path, map_location='cpu')
            model, alphabet = load_model_and_alphabet_core("esm2_t33_650M_UR50D", model_data, None)
            model = model.float()
            _esm2_model_cache = model
            _esm2_alphabet_cache = alphabet
            print(f"[ESM2] ✅ Model cached (1280-dim, 651M params)")
        
        model = _esm2_model_cache.to(device)
        alphabet = _esm2_alphabet_cache
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        # 2️⃣ 编码与推理
        data = [("protein", seq_str)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        # 3️⃣ 提取embedding（去掉BOS/EOS）
        emb = results["representations"][33][0, 1:-1, :]  # [L, 1280]
        
        # 4️⃣ 归一化处理
        emb = F.layer_norm(emb, emb.shape[1:])
        mean = emb.mean()
        std = emb.std()
        emb = (emb - mean) / (std + 1e-6)
        emb = torch.clamp(emb, -5, 5)
        
        # 5️⃣ 显存清理
        emb = emb.detach().cpu()
        del results, batch_tokens
        torch.cuda.empty_cache()
        gc.collect()
        
        return emb
    
    except Exception as e:
        print(f"[ESM2] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return torch.zeros((len(seq_fixed), 1280))

def esm1b_t33_650m(seq, device='cpu'):
    """
    从本地权重加载 ESM-1b t33 650M，提取1280维embedding
    """
    import torch
    import torch.nn.functional as F
    import gc
    import esm
    from esm.pretrained import load_model_and_alphabet_core
    
    global _esm1b_model_cache, _esm1b_alphabet_cache
    
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)
    
    legal_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_fixed = ['A' if aa.upper() not in legal_aa else aa.upper() for aa in seq]
    original_len = len(seq_fixed)
    MAX_LEN = 1022
    if original_len > MAX_LEN:
        seq_fixed = seq_fixed[:MAX_LEN]
        print(f"[ESM1b] Truncated {original_len} -> {MAX_LEN}")
    
    seq_str = ''.join(seq_fixed)
    seq_str = ''.join(seq_fixed)
    
    try:
        # 1️⃣ 模型缓存与加载
        if _esm1b_model_cache is None:
            print(f"[ESM1b] Loading esm1b_t33_650M from local weights...")
            model_path = "/home/ghd/PNBind/GAPointnet_pytorch/esm3/data/weights/esm1b_t33_650M_UR50S.pt"
            model_data = torch.load(model_path, map_location='cpu')
            model, alphabet = load_model_and_alphabet_core("esm1b_t33_650M_UR50S", model_data, None)
            model = model.float()
            _esm1b_model_cache = model
            _esm1b_alphabet_cache = alphabet
            print(f"[ESM1b] ✅ Model cached (1280-dim)")
        
        model = _esm1b_model_cache.to(device)
        alphabet = _esm1b_alphabet_cache
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        # 2️⃣ 编码与推理
        data = [("protein", seq_str)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        # 3️⃣ 提取embedding（去掉BOS/EOS）
        emb = results["representations"][33][0, 1:-1, :]  # [L, 1280]
        
        # 4️⃣ 归一化处理
        emb = F.layer_norm(emb, emb.shape[1:])
        mean = emb.mean()
        std = emb.std()
        emb = (emb - mean) / (std + 1e-6)
        emb = torch.clamp(emb, -5, 5)
        
        # 5️⃣ 显存清理
        emb = emb.detach().cpu()
        del results, batch_tokens
        torch.cuda.empty_cache()
        gc.collect()
        
        return emb
    
    except Exception as e:
        print(f"[ESM1b] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return torch.zeros((len(seq_fixed), 1280))

def esm2_t36_3b(seq, device='cpu'):
    """
    从本地权重加载 ESM-2 t36 3B，提取2560维embedding
    """
    import torch
    import torch.nn.functional as F
    import gc
    import esm
    from esm.pretrained import load_model_and_alphabet_core
    
    global _esm2_3b_model_cache, _esm2_3b_alphabet_cache
    
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)
    
    legal_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_fixed = ['A' if aa.upper() not in legal_aa else aa.upper() for aa in seq]
        # 🔴 新增：截断超长序列（3B模型支持更长）
    original_len = len(seq_fixed)
    MAX_LEN = 2046
    if original_len > MAX_LEN:
        seq_fixed = seq_fixed[:MAX_LEN]
        print(f"[ESM2-3B] Truncated {original_len} -> {MAX_LEN}")
    seq_str = ''.join(seq_fixed)
    
    try:
        # 1️⃣ 模型缓存与加载
        if _esm2_3b_model_cache is None:
            print(f"[ESM2-3B] Loading esm2_t36_3B from local weights...")
            model_path = "/home/ghd/PNBind/GAPointnet_pytorch/esm3/data/weights/esm2_t36_3B_UR50D.pt"
            model_data = torch.load(model_path, map_location='cpu')
            model, alphabet = load_model_and_alphabet_core("esm2_t36_3B_UR50D", model_data, None)
            model = model.float()
            _esm2_3b_model_cache = model
            _esm2_3b_alphabet_cache = alphabet
            print(f"[ESM2-3B] ✅ Model cached (2560-dim, 3B params)")
        
        model = _esm2_3b_model_cache.to(device)
        alphabet = _esm2_3b_alphabet_cache
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        # 2️⃣ 编码与推理
        data = [("protein", seq_str)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[36], return_contacts=False)
        
        # 3️⃣ 提取embedding（去掉BOS/EOS）
        emb = results["representations"][36][0, 1:-1, :]  # [L, 2560]
        
        # 4️⃣ 归一化处理
        emb = F.layer_norm(emb, emb.shape[1:])
        mean = emb.mean()
        std = emb.std()
        emb = (emb - mean) / (std + 1e-6)
        emb = torch.clamp(emb, -5, 5)
        
        # 5️⃣ 显存清理
        emb = emb.detach().cpu()
        del results, batch_tokens
        torch.cuda.empty_cache()
        gc.collect()
        
        return emb
    
    except Exception as e:
        print(f"[ESM2-3B] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return torch.zeros((len(seq_fixed), 2560))   
if __name__ == "__main__":
    lengths = analyze_sequence_lengths()
    configs = [
        ("prna_labels", "prna", "train"),
        ("prna_labels", "prna", "test"),
        ("pdna_labels", "pdna", "train"),
        ("pdna_labels", "pdna", "test"),
    ]

    for root, dataset, split in configs:
        print(f"\n=== 开始生成 {dataset}_{split}.pt ===")
        if dataset == "prna":
            ds = prna(root=root, dataset=dataset, data_split=split)
        else:
            ds = pdna(root=root, dataset=dataset, data_split=split)
        print(f"{dataset}_{split} 样本数: {len(ds)}")

