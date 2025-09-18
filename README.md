# Skeleton Motion Anonymization

This repository contains the official implementation of our paper **"Exposing and Erasing Identity in Skeleton Motion: A New Evaluation Protocol and Adversarial Anonymization Framework."**

---

## 📦 Prerequisites

- **Python** ≥ 3.8  
- **PyTorch** 2.0.1  
- **CUDA** 11.8  

---

## ⚡ Compile CUDA Extensions

```bash
cd ./model/Temporal_shift
bash run.sh
```

---

## 📂 Data Preparation

1. **Download datasets:**
   - [NTU RGB+D 60](https://github.com/shahroudy/NTURGB-D)  
   - [NTU RGB+D 120](https://github.com/shahroudy/NTURGB-D)

2. **Extract skeleton data:**
   - Extract `nturgbd_skeletons_s001_to_s017.zip` into `./data/nturgbd_raw`
   - Extract `nturgbd_skeletons_s018_to_s032.zip` into `./data/nturgbd120star_raw`

3. **Generate joint data:**
```bash
cd data_gen
# For NTU60
python ntu_gendata.py
# For NTU120 (disjoint)
python ntu120disjoint_gendata.py
```

Your final directory structure should look like:

```
data/
├── ntu/
├── ntu120disjoint/
├── nturgbd_raw/
│   ├── S001C001P001R001A001.skeleton
│   └── ...
└── nturgbd120star_raw/
    ├── S018C001P008R001A061.skeleton
    └── ...
```

---

## 🎯 Pre-trained Models

Download checkpoints from [Google Drive](https://drive.google.com/file/d/1FYkf1AX24nuI2INcnv-yxCE3GBK5sDCy/view?usp=sharing) and extract them into:

```
./pretrained/
```

### Test Pre-trained Models
```bash
python main_gan.py --config ./config/test_{task name}.yaml
```

---

## 🏋️ Training

### Baselines (ResNet & U-Net)
```bash
python main_base.py --config ./config/train_adver_resnet.yaml
python main_base.py --config ./config/train_adver_unet.yaml
```

### Proposed Model
```bash
python main_gan.py --config ./config/train_adver_trans_stack_gan.yaml
```

> **Note:** We adopt `main_base.py` from the [original implementation](https://github.com/ml-postech/Skeleton-anonymization) and fix the missing `model.eval()` call during evaluation.

---

## 🙏 Acknowledgements

This code is based on:  
- [Skeleton-anonymization](https://github.com/ml-postech/Skeleton-anonymization)  
- [Shift-GCN](https://github.com/kchengiva/Shift-GCN)  
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)  
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)  
- [MAMP](https://github.com/maoyunyao/MAMP)  
