# Transparency-guided-EnsembleCNN
This repository holds the Keras code for the paper

**Transparency-guided ensemble convolutional neural network for the stratification between pseudoprogression and true progression of glioblastoma multiform in MRI** 

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

# Author List
Xiaoming Liu, Xiaobo Zhou, Xiaohua Qian*

# Abstract
For patients with glioblastoma multiform (GBM), differentiating pseudoprogression (PsP) from true tumor progression (TTP) is a challenging and time-consuming task for radiologists. Although deep neural networks can automatically diagnose PsP and TTP, lacking of interpretability has always been its major drawback. To overcome these shortcomings and produce more reliable outcomes, we propose a transparency-guided ensemble convolutional neural network (CNN) to automatically discriminate PsP and TTP in magnetic resonance imaging (MRI). A total of 84 patients with GBM were enrolled in the study. First, three typical convolutional neutral networks, namely VGG, ResNet and DenseNet, were trained to distinguish PsP and TTP. Subsequently, we used class-specific gradient information from convolutional layers to highlight the important regions in MRI scans. And radiologists selected the most lesion-relevant layer for each CNN. Finally, the selected layers are utilized to guide the construction of a multi-scale ensemble CNN whose classification accuracy reached 90.20%, and whose specificity is promoted 20% than that of a single CNN. The results demonstrate the presented network can enhance the reliability and accuracy of CNNs.

# Requied
Our code is based on **Python**.

# Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = {Transparency-guided ensemble convolutional neural network for the stratification between pseudoprogression and true progression of glioblastoma multiform in MRI},
  author    = {Xiaoming Liu, Xiaobo Zhou, Xiaohua Qian*},
  journal   = {Journal of Visual Communication and Image Representation},
  month     = {October}，
  year      = {2020},
}
```

# Contact
For any question, feel free to contact
```
Xiaohua Qian: xiaohua.qian@sjtu.edu.cn
```
