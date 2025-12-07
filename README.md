# Towards Attention-Aware Person Re-Identification

This repository contains the official implementation of **AaP-ReID**, a person re-identification architecture that enhances identity-discriminative learning through **Channel Attention Bottlenecks (CAB)** and **shortest-path part-aware feature aggregation**. The method improves discrimination in deeper layers while suppressing background activation.

---

## 📘 Architecture Overview

<p align="center">
  <img src="/models/Architecture-updated.png" width="650">
</p>

*(The architecture image is included in this repository under `assets/`. You may adjust the path above if needed.)*

---

## 🔗 Pretrained Weights

Model checkpoints are hosted on Google Drive:

👉 **Download Weights:**  
https://drive.google.com/drive/folders/XXXXXXXXXXXX  

The Drive contains the following folders:
cuhk03/
dukemtmcreid/
market1501/
msmt17/


### ➤ Place the weights inside: 
log/AaP-ReID

## 📂 Dataset Setup

Supported datasets:

- Market1501  
- DukeMTMC-reID  
- CUHK03  
- MSMT17  

Download each dataset manually and store them in:
/data

## 🛠 Installation

Install all required Python packages:

```bash
pip install -r requirements.txt

### ➤ Train: 
python Train.py --config configs/train_market1501.yaml

### ➤ Eval: 
python Train.py --config configs/train_market1501.yaml


## Acknowledgements
This implementation is built upon the AlignedReID++ framework  
(https://github.com/michuanhaohao/AlignedReID).


