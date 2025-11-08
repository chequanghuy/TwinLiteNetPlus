
<h1> TwinLiteNet+: An Enhanced Multi-Task Segmentation Model for Autonomous Driving
 </h1>



[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/chequanghuy/TwinLiteNetPlus/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.8-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.8-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

  [Quang-Huy Che](https://scholar.google.com/citations?user=k7lUdFAAAAAJ&hl=vi&authuser=2), [Duc-Tri Le](https://github.com/DucTriCE?tab=repositories), Minh-Quan Pham, Vinh-Tiep Nguyen, Duc-Khai Lam
</div>




## 📢 Publication

We are pleased to announce that our paper has been **accepted for publication** in the journal *Computers and Electrical Engineering* (Elsevier).

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Semantic segmentation is crucial for autonomous driving, particularly for Drivable Area and Lane Segmentation, ensuring safety and navigation. To address the high computational costs of current state-of-the-art (SOTA) models, this paper introduces TwinLiteNetPlus (TwinLiteNet+), a model adept at balancing efficiency and accuracy. TwinLiteNet+ incorporates standard and depth-wise separable dilated convolutions, reducing complexity while maintaining high accuracy. It is available in four configurations, from the robust 1.94 million-parameter TwinLiteNetPlus_Large to the ultra-compact 34K-parameter TwinLiteNetPlus_nano. Notably, TwinLiteNetPlus_Large attains a 92.9% mIoU for Drivable Area Segmentation and a 34.2% IoU for Lane Segmentation. These results notably outperform those of current SOTA models while requiring a computational cost that is approximately 11 times lower in terms of Floating Point Operations (FLOPs) compared to the existing SOTA model. Extensively tested on various embedded devices, TwinLiteNet+ demonstrates promising latency and power efficiency, underscoring its suitability for real-world autonomous vehicle applications.
</details>


<p align="center">
  <img src="https://github.com/user-attachments/assets/33418dc9-50a6-4c4b-8206-b1725d2f034e" width=90%> <br>

</p>

## Main Results

<p align="left">
  <img src="https://github.com/user-attachments/assets/fd17c313-ff9a-4044-a9c5-8c15b482606b" width=50%> <br>
  Comparison of evaluation metrics mIoU (Drivable Area Segmentation) - IoU (Lane Segmentation) - GFLOPs of various models on the BDD100K dataset.
</p>





| Model | Drivable Area mIoU (%) ↑ | Lane Accuracy (%) ↑ | Lane IoU (%) ↑ | FLOPS ↓ | #Params ↓ |
|--------|----------------|----------------|--------------|----------------|----------------|
| DeepLabV3+ | 90.9 | -- | 29.8 | 30.7G | 15.4M |
| SegForme | 92.3 | -- | 31.7 | 12.1G | 7.2M |
| R-CNNP | 90.2 | -- | 24.0 | -- | -- |
| YOLOP | 91.6 | -- | 26.5 | 8.11G | 5.53M |
| IALaneNet (ResNet-18) | 90.54 | -- | 30.39 | 89.83G | 17.05M |
| IALaneNet (ResNet-34) | 90.61 | -- | 30.46 | 139.46G | 27.16M |
| IALaneNet (ConvNeXt-tiny) | 91.29 | -- | 31.48 | 96.52G | 18.35M |
| IALaneNet (ConvNeXt-small) | 91.72 | -- | 32.53 | 200.07G | 39.97M |
| YOLOv8 (multi) | 84.2 | 81.7 | 24.3 | -- | -- |
| Sparse U-PDP | 91.5 | -- | 31.2 | -- | -- |
| TwinLiteNet | 91.3 | 77.8 | 31.1 | 3.9G | 0.44M |
| **TwinLiteNet+ Nano** | 87.3 | 70.2 | 23.3 | **0.57G** | **0.03M** |
| **TwinLiteNet+ Small** | 90.6 | 75.8 | 29.3 | 1.40G | 0.12M |
| **TwinLiteNet+ Medium** | 92.0 | 79.1 | 32.3 | 4.63G | 0.48M |
| **TwinLiteNet+ Large** | **92.9** | **81.9** | **34.2** | 17.58G | 1.94M |

**Notes:**
- ↑ indicates higher values are better.
- ↓ indicates lower values are better.
- "--" indicates unavailable values.


## Requirement

This codebase has been developed with python version 3..8, PyTorch 1.8.0 and torchvision 0.9.0
```setup
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or
```setup
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```
See `requirements.txt` for additional dependencies and version requirements.
```setup
pip install -r requirements.txt
```


## Pre-trained Model
You can get the pre-trained model from <a href="https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz?usp=sharing">google</a>.


## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [drivable_are_annotations](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing), [lane_line_annotations](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing)

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─bdd100k
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─drivable_are_annotations
│ │ ├─train
│ │ ├─val
│ ├─lane_line_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./BDD100K.py`.

## Training

### Multi-task

```shell
python train.py --config '{nano/small/medium/large}'
```

### Single-task

```shell
python train_singletask.py --config '{nano/small/medium/large}' --task '{"DA"/"LL"}'  # DA for drivable area, LL for lane line 
```

## Evaluation

```shell
python val.py --config '{nano/small/medium/large}' --weight 'pretrained/{nano/small/medium/large}.pth'
```

## Demo

```shell
python demo.py --config '{nano/small/medium/large}' --weight 'pretrained/{nano/small/medium/large}.pth' --source 'pretrained/{images/videos}'
```


## License

TwinLiteNetPlus is released under the [MIT Licence](LICENSE).

## Acknowledgements



* [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
* [Partial Class Activation Attention for Semantic Segmentation](https://github.com/lsa1997/PCAA)
* [ESPNet](https://github.com/sacmehta/ESPNet)

## Citation

```BibTeX
@article{CHE2025110694,
     title = {TwinLiteNet+: An enhanced multi-task segmentation model for autonomous driving},
     journal = {Computers and Electrical Engineering},
     volume = {128},
     pages = {110694},
     year = {2025},
     issn = {0045-7906},
     doi = {https://doi.org/10.1016/j.compeleceng.2025.110694},
     url = {https://www.sciencedirect.com/science/article/pii/S0045790625006378}
}
```

