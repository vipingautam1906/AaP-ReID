# Towards Attention-Aware Person Re-Identification



<p align="center">
  <img src="/models/Architecture-updated.png" width="650">
</p>



---

## 🔗 Pretrained Weights

Model checkpoints are hosted on Google Drive:

**Download Weights:**  
https://drive.google.com/drive/folders/1ZYfZHCmOmi52udcGZgKKNBuPCMuGX8ra?usp=sharing  

The drive contains the following folders:

<br>cuhk03/
<br>dukemtmcreid/
<br>market1501/
<br>msmt17/


**Place the weights inside** 
<br>log/AaP-ReID
<br>The directory tree should look like:

<pre>
  log/
└── AaP-ReID/
    ├── market1501/
    │   └── model_best.pth
    ├── dukemtmcreid/
    │   └── model_best.pth
    ├── cuhk03/
    │   └── model_best.pth
    └── msmt17/
        └── model_best.pth
</pre>


## 🔗 Dataset Setup

**Supported datasets**

- Market1501  
- DukeMTMC-reID  
- CUHK03  
- MSMT17  

**Download Datasets**  
https://drive.google.com/drive/folders/1ZYfZHCmOmi52udcGZgKKNBuPCMuGX8ra?usp=sharing  

Store the reID datasets under /data <br> 
cd AaP-ReID/data

<pre>
  data/
├── market1501/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   ├── query/
│   └── ... 
│
├── dukemtmcreid/
│   ├── DukeMTMC-reID/
│   │   ├── bounding_box_train/
│   │   ├── bounding_box_test/
│   │   ├── query/
│   │   └── ...
│   └── ...
│
├── cuhk03/
│   ├── images/
│   ├── cuhk03_new_protocol_config_detected.mat
│   └── ...
│
└── msmt17/
    ├── train/
    ├── test/
    ├── list_train.txt
    ├── list_val.txt
    ├── list_query.txt
    ├── list_gallery.txt
    └── ...

</pre>

## 🔗 Installation

Install all required Python packages:


<pre>
  pip install -r requirements.txt
</pre>

## 🔗 Train and Evaluation
to change dataset use (-d cuhk03, -d msmt17, -d dukemtmcreid)

**Train** 
<pre>
  python3 Train.py  -d market1501 -a resnet50 --test_distance global_local --labelsmooth
</pre>


**Eval** 
<pre>
  python3 Train.py -d market1501 -a resnet50 --evaluate --resume /AaP-ReID/market1501/best_model.pth.tar --save-dir /AaP-ReID/log/eval --test_distance global_local --labelsmooth
</pre>




