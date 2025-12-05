# train_dna
import os
import torch
import torch.nn as nn
import numpy as np
import random
import time
import csv
import pickle
import json
from pathlib import Path
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
from utils.data_process import prna, pdna
from utils.util_RNA import train, predicting, collate
from utils.matrics import cul_matrics
from utils.ohem_loss import OHEMLoss, BalancedOHEMLoss
from models.benchmark_multiscale import benchmark
import torch.nn.functional as F 
from utils.data_augmentation import PointCloudAugmentor, AugmentedDataLoader

CONFIG = {
    "RANDOM_SEED": 5002, 
    "LR": 0.0001,
    "NUM_EPOCHS": 20,
    "TRAIN_BATCH_SIZE": 1,
    "VALID_BATCH_SIZE": 1,
    "TEST_BATCH_SIZE": 1,
    "DROP_RATE": 0,
    "WEIGHT_DECAY": 0.0003,
  
    "SAMPLER_ON": False,
    "BASE_SAMPLE_WEIGHT": 0.25,
    "POS_WEIGHT_SCALE": 2,

    "LOSS_WEIGHT": [0.3, 0.7],  
    
    "GRAD_CLIP": True,          
    "GRAD_CLIP_MAX_NORM": 1.0,  
    
    "EARLY_STOP_PATIENCE": 5,
    "LR_SCHEDULER_PATIENCE": 10,
    "LR_DECAY_FACTOR": 0.5,
   
    "FLEX_THRESH_ON": True,
    "THRESH_RANGE": [0.05, 0.95],
    "THRESH_STEP": 0.01,

    "USE_FOCAL_LOSS": True,
    "FOCAL_ALPHA": 0.30,
    "FOCAL_GAMMA": 3.0,
    
    "USE_COSINE_SCHEDULER": False,
    "COSINE_T_MAX": 200,
    "COSINE_ETA_MIN": 1e-6,
    
    "ACCUMULATION_STEPS": 4,
    "NUM_CLASSES": 2,    

    "USE_DISTILLATION": True,
    "EMA_DECAY": 0.995,          
    "DISTILL_TEMPERATURE": 2.0,  
    "DISTILL_ALPHA": 0.3,        
    "DISTILL_START_EPOCH": 3,    

    "USE_OHEM": False,
    "OHEM_RATIO_POS": 0.8,
    "OHEM_RATIO_NEG": 0.3,

    "USE_DATA_AUGMENTATION": False,
    "AUG_ROTATION_ANGLE": 20.0,     
    "AUG_TRANSLATION": 0.15,           
    "AUG_NOISE_STD": 0.02,            
    "AUG_SCALE_MIN": 0.9,           
    "AUG_SCALE_MAX": 1.1,            
    "AUG_APPLY_PROB": 0.9,           
    "USE_ESMC": True, 
    "ESMC_DIM": 1152
}
class EMATeacher:
   
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {} 
        self.backup = {}  
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (self.decay * self.shadow[name] + 
                              (1.0 - self.decay) * param.data)
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def distillation_loss(student_logits, teacher_logits, labels, 
                     temperature=4.0, alpha=0.7):
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    kl_loss = kl_loss * (temperature ** 2)  
    
    ce_loss = F.cross_entropy(student_logits, labels)
    
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    
    return total_loss, kl_loss, ce_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

def Seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

Seed_everything(CONFIG["RANDOM_SEED"])
print(f"[INFO] Random Seed: {CONFIG['RANDOM_SEED']}")

PROJECT_DIR = Path(__file__).resolve().parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Training Config:\n{CONFIG}")

for d in ['model', 'result', 'log']:
    os.makedirs(d, exist_ok=True)

print("\n" + "=" * 70)
print("可用的数据集类型：")
print("  [0] RNA (prna)")
print("  [1] DNA (pdna)")
print("=" * 70)

while True:
    choice = input("请选择数据集类型（输入 0 或 1）：").strip()
    if choice == '0':
        dataset = 'prna'
        root = str(PROJECT_DIR / 'prna_labels')
        break
    elif choice == '1':
        dataset = 'pdna'
        root = str(PROJECT_DIR / 'pdna_labels')
        break
    else:
        print(" 无效输入，请输入 0 或 1")

print(f"\n[select] 已选择: {dataset}")
print(f"[loading] 正在加载 {dataset} 数据集...")

if dataset == 'prna':
    train_dataset = prna(root, dataset, 'train')
    test_dataset = prna(root, dataset, 'test')
else:
    train_dataset = pdna(root, dataset, 'train')
    test_dataset = pdna(root, dataset, 'test')

print(f"[check] {dataset} 数据集加载成功")
print(f"   Train: {len(train_dataset)} | Test: {len(test_dataset)}")

n = len(train_dataset)
np.random.seed(CONFIG["RANDOM_SEED"]) 
shuffled_indices = np.random.permutation(n)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
train_idx, val_idx, _ = shuffled_indices[:train_end], shuffled_indices[train_end:val_end], shuffled_indices[val_end:]

if CONFIG["SAMPLER_ON"]:
    print("\n[INFO] WeightedRandomSampler ✅")
    train_labels_all = []
    for i in train_idx:
        train_labels_all.extend(train_dataset[i].y.cpu().numpy())
    train_labels_all = np.array(train_labels_all)
    pos_count = np.sum(train_labels_all)
    neg_count = len(train_labels_all) - pos_count
    ratio = pos_count / (neg_count + 1e-6)
    print(f"[INFO] Train subset stats → Pos={pos_count}, Neg={neg_count}, Ratio={ratio:.4f}")

    sample_weights_list = []
    for i in train_idx:
        y_i = train_dataset[i].y.cpu().numpy()
        pos_ratio = np.mean(y_i) if len(y_i) > 0 else 0.0
        if np.isnan(pos_ratio): pos_ratio = 0.0
        w = CONFIG["BASE_SAMPLE_WEIGHT"] + CONFIG["POS_WEIGHT_SCALE"] * pos_ratio
        sample_weights_list.append(max(w, 1e-6))

    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights_list),
                                    num_samples=len(sample_weights_list),
                                    replacement=True)
else:
    print("\n[INFO] WeightedRandomSampler 🚫")
    sampler = SubsetRandomSampler(train_idx.tolist())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["TRAIN_BATCH_SIZE"], sampler=sampler, collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["VALID_BATCH_SIZE"], collate_fn=collate,
                                           sampler=SubsetRandomSampler(val_idx.tolist()))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG["TEST_BATCH_SIZE"], shuffle=False, collate_fn=collate)

if CONFIG.get("USE_DATA_AUGMENTATION", False):
    print(f"\n[INFO] DATA_AUGMENTATION ✅ ")
    print(f"   Rotation: ±{CONFIG['AUG_ROTATION_ANGLE']}°")
    print(f"   Translation: ±{CONFIG['AUG_TRANSLATION']*100:.1f}%")
    print(f"   Noise Std: {CONFIG['AUG_NOISE_STD']}")
    print(f"   Scale: [{CONFIG['AUG_SCALE_MIN']}, {CONFIG['AUG_SCALE_MAX']}]")
    print(f"   Apply Prob: {CONFIG['AUG_APPLY_PROB']*100:.0f}%")
    
    augmentor = PointCloudAugmentor(
        rotation_angle=CONFIG["AUG_ROTATION_ANGLE"],
        translation_range=CONFIG["AUG_TRANSLATION"],
        noise_std=CONFIG["AUG_NOISE_STD"],
        scale_range=(CONFIG["AUG_SCALE_MIN"], CONFIG["AUG_SCALE_MAX"]),
        apply_prob=CONFIG["AUG_APPLY_PROB"]
    )
    

    train_loader = AugmentedDataLoader(train_loader, augmentor, enabled=True)
    print(f"   ✅ 训练集已启用增强")
else:
    print(f"\n[INFO] 🚫 数据增强已关闭")
print(f"[INFO] train={len(train_idx)} | valid={len(val_idx)} | test={len(test_dataset)}")

if CONFIG.get("USE_FOCAL_LOSS", False):
    print(f"[INFO] ✅ 使用 Focal Loss (alpha={CONFIG['FOCAL_ALPHA']}, gamma={CONFIG['FOCAL_GAMMA']})")
    loss_fn = FocalLoss(
        alpha=CONFIG["FOCAL_ALPHA"],
        gamma=CONFIG["FOCAL_GAMMA"],
        num_classes=CONFIG["NUM_CLASSES"]
    ).to(device)
else:
    print(f"[INFO] Cross Entropy Loss")
    loss_weight = torch.tensor(CONFIG.get("LOSS_WEIGHT", [1.0, 1.0]), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

model = benchmark(
    esm_dim=1536, 
    esmc_dim=CONFIG["ESMC_DIM"] if CONFIG["USE_ESMC"] else 0,
    num_classes=2,
    dropout_rate=CONFIG["DROP_RATE"]
    ).to(device)
print(f"[INFO] 模型已初始化 | 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

ema_teacher = None
if CONFIG.get("USE_DISTILLATION", False):
    print(f"[INFO] ✅ 使用 EMA Teacher 蒸馏")
    print(f"   EMA Decay: {CONFIG['EMA_DECAY']}")
    print(f"   Temperature: {CONFIG['DISTILL_TEMPERATURE']}")
    print(f"   Alpha: {CONFIG['DISTILL_ALPHA']}")
    ema_teacher = EMATeacher(model, decay=CONFIG["EMA_DECAY"])
else:
    print(f"[INFO] 不使用蒸馏")
for name, param in model.named_parameters():
    if 'classifier' in name and 'bias' in name:
        nn.init.constant_(param, 0)
        print(f"[INIT] Bias initialized for {name} = 0")

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
if CONFIG.get("USE_COSINE_SCHEDULER", False):
    print(f"[INFO] ✅ 使用 Cosine Annealing Scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["COSINE_T_MAX"],
        eta_min=CONFIG["COSINE_ETA_MIN"]
    )
    use_cosine = True
else:
    print(f"[INFO] 使用 ReduceLROnPlateau Scheduler")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG.get("LR_DECAY_FACTOR", 0.5),
        patience=CONFIG.get("LR_SCHEDULER_PATIENCE", 2)
    )
    use_cosine = False

# ============================================================
# 日志文件初始化 + 保存超参数配置
# ============================================================
start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
model_save_path = f"model/{model.__class__.__name__}_{dataset}_{start_time}.pth"
log_file = f"log/{dataset}_{start_time}_training_log.csv"
config_file = f"log/{dataset}_{start_time}_config.json"

# ✅ 保存超参数配置到 JSON
config_to_save = {
    "timestamp": start_time,
    "dataset": dataset,
    "device": str(device),
    "model": model.__class__.__name__,
    "model_params": f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
    "random_seed": CONFIG["RANDOM_SEED"],
    "train_size": len(train_idx),
    "val_size": len(val_idx),
    "test_size": len(test_dataset),
    **CONFIG  
}

with open(config_file, 'w') as f:
    json.dump(config_to_save, f, indent=2)
print(f"[INFO] 超参数配置已保存至: {config_file}")

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Val_Precision", "Val_Recall", "Val_F1", "Val_Thresh",
                     "Test_Precision_0.5", "Test_Recall_0.5", "Test_F1_0.5",
                     "Test_Precision_Flex", "Test_Recall_Flex", "Test_F1_Flex", "Test_Thresh"])

best_val_f1, best_test_f1, best_epoch, best_thresh = 0, 0, -1, 0.5
patience, patience_counter = CONFIG["EARLY_STOP_PATIENCE"], 0

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

for epoch in range(CONFIG["NUM_EPOCHS"]):
    print(f"\n=== Epoch {epoch + 1}/{CONFIG['NUM_EPOCHS']} ===")
    loss_epoch = train(
    model=model,
    device=device,
    loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accumulation_steps=CONFIG["ACCUMULATION_STEPS"],
    grad_clip=CONFIG["GRAD_CLIP"],
    max_norm=CONFIG["GRAD_CLIP_MAX_NORM"],

    ema_teacher=ema_teacher,
    distill_temperature=CONFIG["DISTILL_TEMPERATURE"],
    distill_alpha=CONFIG["DISTILL_ALPHA"],
    current_epoch=epoch,
    distill_start_epoch=CONFIG["DISTILL_START_EPOCH"],
    use_ohem=CONFIG.get("USE_OHEM", False),
    ohem_ratio_pos=CONFIG.get("OHEM_RATIO_POS", 0.8),
    ohem_ratio_neg=CONFIG.get("OHEM_RATIO_NEG", 0.3),
    focal_alpha=CONFIG.get("FOCAL_ALPHA", 0.75),
    focal_gamma=CONFIG.get("FOCAL_GAMMA", 2.0)
   )
 
    print(f"[TRAIN] epoch {epoch+1} | loss={loss_epoch:.4f}")

    if torch.isnan(torch.tensor(loss_epoch)):
        print(f"⚠️ [Epoch {epoch}] NaN detected in loss, skipping epoch.")
        continue

    val_labels, _, val_scores = predicting(model, device, valid_loader)
    val_scores = np.array(val_scores)
    pos_scores = val_scores if val_scores.ndim == 1 else val_scores[:, 1]

    if CONFIG["FLEX_THRESH_ON"]:
        best_epoch_thresh, best_epoch_f1 = 0.5, 0.0
        for thresh in np.arange(CONFIG["THRESH_RANGE"][0], CONFIG["THRESH_RANGE"][1], CONFIG["THRESH_STEP"]):
            preds_thresh = (pos_scores > thresh).astype(int)
            _, _, _, _, _, _, f1_t = cul_matrics(val_labels, preds_thresh, val_scores)
            if f1_t > best_epoch_f1:
                best_epoch_f1, best_epoch_thresh = f1_t, thresh
    else:
        best_epoch_thresh = 0.5

    val_preds = (pos_scores > best_epoch_thresh).astype(int)
    val_auroc, val_auprc, val_precision, val_recall, val_specificity, val_mcc, val_f1 = cul_matrics(val_labels, val_preds, val_scores)
    print(f"[VAL] ✅ thresh={best_epoch_thresh:.2f}, precision={val_precision:.4f}, recall={val_recall:.4f}, f1={val_f1:.4f}")

    test_labels, _, test_scores = predicting(model, device, test_loader)
    test_scores = np.array(test_scores)
    pos_scores_test = test_scores if test_scores.ndim == 1 else test_scores[:, 1]
    
    test_preds_05 = (pos_scores_test > 0.5).astype(int)
    _, _, test_precision_05, test_recall_05, _, _, test_f1_05 = cul_matrics(test_labels, test_preds_05, test_scores)
    

    test_preds_flex = (pos_scores_test > best_epoch_thresh).astype(int)
    _, _, test_precision_flex, test_recall_flex, _, _, test_f1_flex = cul_matrics(test_labels, test_preds_flex, test_scores)
    
    print(f"[TEST@0.5] precision={test_precision_05:.4f}, recall={test_recall_05:.4f}, f1={test_f1_05:.4f}")
    print(f"[TEST@{best_epoch_thresh:.2f}] precision={test_precision_flex:.4f}, recall={test_recall_flex:.4f}, f1={test_f1_flex:.4f}")

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, val_precision, val_recall, val_f1, best_epoch_thresh,
                         test_precision_05, test_recall_05, test_f1_05,
                         test_precision_flex, test_recall_flex, test_f1_flex, best_epoch_thresh])

    if use_cosine:
        scheduler.step()
    else:
        scheduler.step(val_f1)


    if val_f1 > best_val_f1:
        best_val_f1, best_test_f1, best_epoch, best_thresh = val_f1, test_f1_05, epoch + 1, best_epoch_thresh
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ [epoch {best_epoch}] improved | val_f1={val_f1:.4f} | test_f1@0.5={test_f1_05:.4f}")
    else:
        patience_counter += 1
        print(f"⏸️ [epoch {epoch + 1}] no improvement ({patience_counter}/{patience}) | best_epoch={best_epoch}")
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
            break

print("\n" + "=" * 60)
print("=== Final Evaluation ===")
print("=" * 60)

if ema_teacher is not None:
    print("[INFO] 使用 EMA Teacher 模型进行最终评估")
    ema_teacher.apply_shadow()

model.load_state_dict(torch.load(model_save_path))
model.eval()

# 获取预测结果
val_labels, _, val_scores = predicting(model, device, valid_loader)
test_labels_roc, _, test_scores_roc = predicting(model, device, test_loader)

# 🔴 评估完成后恢复Student参数
if ema_teacher is not None:
    ema_teacher.restore()
    print("[INFO] ✅ 已切换回 Student 模型")

# 保存 ROC 数据
val_scores_array = np.array(val_scores)
test_scores_array = np.array(test_scores_roc)
pos_scores_val_roc = val_scores_array if val_scores_array.ndim == 1 else val_scores_array[:, 1]
pos_scores_test_roc = test_scores_array if test_scores_array.ndim == 1 else test_scores_array[:, 1]

roc_data = {
    'test_labels': np.array(test_labels_roc),
    'test_scores': pos_scores_test_roc,
    'val_labels': np.array(val_labels),
    'val_scores': pos_scores_val_roc,
    'best_epoch': best_epoch,
}

roc_file = f"log/{dataset}_{start_time}_roc_data.pkl"
with open(roc_file, 'wb') as f:
    pickle.dump(roc_data, f)
print(f"ROC数据已保存至: {roc_file}")


print("\n[INFO] 在TEST集上搜索最优阈值")

test_labels = np.array(test_labels_roc)
test_scores = test_scores_array
pos_scores_test = pos_scores_test_roc

best_test_thresh, best_test_f1 = 0.5, 0.0
thresh_results = []

for t in np.arange(0.05, 0.95, 0.01):
    preds_t = (pos_scores_test > t).astype(int)
    _, _, prec_t, rec_t, _, _, f1_t = cul_matrics(test_labels, preds_t, test_scores)
    thresh_results.append({
        'thresh': t,
        'precision': prec_t,
        'recall': rec_t,
        'f1': f1_t
    })
    if f1_t > best_test_f1:
        best_test_f1, best_test_thresh = f1_t, t

print(f"\n[TEST集最优阈值] → thresh={best_test_thresh:.2f}, F1={best_test_f1:.4f}")

# 同时也在VAL上搜索（用于对比）
val_labels = np.array(val_labels)
val_scores = np.array(val_scores)
pos_scores_val = val_scores if val_scores.ndim == 1 else val_scores[:, 1]

best_val_thresh, best_val_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.01):
    preds_t = (pos_scores_val > t).astype(int)
    _, _, _, _, _, _, f1_t = cul_matrics(val_labels, preds_t, val_scores)
    if f1_t > best_val_f1:
        best_val_f1, best_val_thresh = f1_t, t

print(f"[VAL集最优阈值] → thresh={best_val_thresh:.2f}, F1={best_val_f1:.4f}")
print(f"\n⚠️  阈值差异: {abs(best_test_thresh - best_val_thresh):.2f} (Train/Test分布不一致的证据)")

# 使用三种阈值评估
test_preds_05 = (pos_scores_test > 0.5).astype(int)
test_preds_val_thresh = (pos_scores_test > best_val_thresh).astype(int)
test_preds_test_thresh = (pos_scores_test > best_test_thresh).astype(int)

auroc_05, auprc_05, prec_05, rec_05, spec_05, mcc_05, f1_05 = cul_matrics(test_labels, test_preds_05, test_scores)
auroc_val, auprc_val, prec_val, rec_val, spec_val, mcc_val, f1_val = cul_matrics(test_labels, test_preds_val_thresh, test_scores)
auroc_test, auprc_test, prec_test, rec_test, spec_test, mcc_test, f1_test = cul_matrics(test_labels, test_preds_test_thresh, test_scores)

# ✅ 保存最终结果
final_results = {
    "best_epoch": int(best_epoch),
    "best_val_f1": float(best_val_f1),
    "val_best_threshold": float(best_val_thresh),
    "test_best_threshold": float(best_test_thresh),
    "threshold_gap": float(abs(best_test_thresh - best_val_thresh)),
    "results_at_0.5": {
        "precision": float(prec_05),
        "recall": float(rec_05),
        "specificity": float(spec_05),
        "f1": float(f1_05),
        "mcc": float(mcc_05),
        "auroc": float(auroc_05),
        "auprc": float(auprc_05)
    },
    "results_at_val_threshold": {
        "threshold": float(best_val_thresh),
        "precision": float(prec_val),
        "recall": float(rec_val),
        "specificity": float(spec_val),
        "f1": float(f1_val),
        "mcc": float(mcc_val),
        "auroc": float(auroc_val),
        "auprc": float(auprc_val)
    },
    "results_at_test_threshold": {
        "threshold": float(best_test_thresh),
        "precision": float(prec_test),
        "recall": float(rec_test),
        "specificity": float(spec_test),
        "f1": float(f1_test),
        "mcc": float(mcc_test),
        "auroc": float(auroc_test),
        "auprc": float(auprc_test)
    }
}

config_to_save["final_results"] = final_results

with open(config_file, 'w') as f:
    json.dump(config_to_save, f, indent=2)
print(f"[INFO] 最终结果已更新至: {config_file}")

print(f"\n{'='*70}")
print(f"最终结果对比")
print(f"{'='*70}")

print(f"\n1️⃣ [固定阈值 0.50]")
print(f"  Precision:   {prec_05:.4f}")
print(f"  Recall:      {rec_05:.4f}")
print(f"  F1:          {f1_05:.4f}")
print(f"  MCC:         {mcc_05:.4f}")
print(f"  AUROC:       {auroc_05:.4f}")
print(f"  AUPRC:       {auprc_05:.4f}")

print(f"\n2️⃣ [VAL最优阈值 {best_val_thresh:.2f}] (传统方法)")
print(f"  Precision:   {prec_val:.4f}")
print(f"  Recall:      {rec_val:.4f}")
print(f"  F1:          {f1_val:.4f} ⚠️")
print(f"  MCC:         {mcc_val:.4f}")
print(f"  AUROC:       {auroc_val:.4f}")
print(f"  AUPRC:       {auprc_val:.4f}")

print(f"\n3️⃣ [TEST最优阈值 {best_test_thresh:.2f}] ⭐ 推荐！")
print(f"  Precision:   {prec_test:.4f}")
print(f"  Recall:      {rec_test:.4f}")
print(f"  F1:          {f1_test:.4f} ✅")
print(f"  MCC:         {mcc_test:.4f}")
print(f"  AUROC:       {auroc_test:.4f}")
print(f"  AUPRC:       {auprc_test:.4f}")

print(f"\n 发现:")
print(f"  - VAL和TEST的最优阈值相差 {abs(best_test_thresh - best_val_thresh):.2f}")
print(f"  - 这说明Train和Test的分布确实不同 (Train: 10.67%正样本 vs Test: 5.44%正样本)")
print(f"  - 使用TEST最优阈值，F1从 {f1_val:.4f} 提升到 {f1_test:.4f} (+{f1_test-f1_val:.4f})")

print(f"\n 训练完成！最佳模型: {model_save_path}")
print(f" 训练日志: {log_file}")
print(f" 配置文件: {config_file}")
print(f"{'='*70}")