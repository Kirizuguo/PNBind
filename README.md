# PNBind

Prediction of Nucleic Acid-Binding Sites Using Protein Structure and Protein Language Models

## Installation
```bash
git clone https://github.com/Kirizuguo/PNBind.git
cd PNBind
pip install -r requirements.txt
```

## Download ESM Models

Download the following model weights and place them in `weights/` directory:

| Model | Download Link |
|-------|---------------|
| ESM3-sm-open-v1 | https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1 |
| ESMC-600M | https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12 |
| ESM2-650M | https://huggingface.co/facebook/esm2_t33_650M_UR50D |
| ESM2-3B | https://huggingface.co/facebook/esm2_t36_3B_UR50D |

## Data

- **DNA-129**: 573 training + 129 test protein chains
- **RNA-117**: 495 training + 117 test protein chains

Source: [BioLiP database](https://zhanggroup.org/BioLiP/)

## Usage
```bash
python train.py --dataset pdna --data_split train
```

## Citation
```bibtex
@article{li2025pnbind,
  title={PNBind: Prediction of Nucleic Acid-Binding Sites Using Protein Structure and Protein Language Models},
  author={Li, Yunhai and Dang, Guanghong and Shao, Bowen and Li, Pengpai and Liu, Zhi-Ping},
  journal={Journal of Chemical Information and Modeling},
  year={2025}
}
```

## Contact

zpliu@sdu.edu.cn
