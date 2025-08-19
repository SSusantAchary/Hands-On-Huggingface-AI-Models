# Core Computer Vision Models Overview
## Classification Models

| **Model**                    | **Task**             | **Code / Weights**                                                                                         | **Paper**                                                                | **License**             |
| ---------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------- |
| **AlexNet**                  | Image Classification | [Cuda-ConvNet2 (Alex Krizhevsky)](https://github.com/akrizhevsky/cuda-convnet2)                            | *ImageNet Classification with Deep CNNs* (NeurIPS 2012)                  | Apache 2.0              |
| **VGG** (VGG16)              | Image Classification | [Oxford VGG Model Release](https://www.robots.ox.ac.uk/~vgg/research/very_deep/)                           | *Very Deep Conv. Networks for Large-Scale Image Recognition* (ICLR 2015) | Creative Commons BY 4.0 |
| **ResNet**                   | Image Classification | [Kaiming He’s ResNet Repo](https://github.com/KaimingHe/deep-residual-networks)                            | *Deep Residual Learning for Image Recognition* (CVPR 2016)               | MIT                     |
| **DenseNet**                 | Image Classification | [DenseNet Official (Torch)](https://github.com/liuzhuang13/DenseNet)                                       | *Densely Connected Convolutional Networks* (CVPR 2017)                   | BSD 3-Clause            |
| **EfficientNet**             | Image Classification | [TensorFlow TPU EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)  | *EfficientNet: Rethinking Model Scaling* (ICML 2019)                     | Apache 2.0              |
| **Vision Transformer (ViT)** | Image Classification | [Google Research ViT](https://github.com/google-research/vision_transformer)                               | *An Image is Worth 16x16 Words* (ICLR 2021)                              | Apache 2.0              |
| **Swin Transformer**         | Image Classification | [Microsoft Swin-Transformer](https://github.com/microsoft/Swin-Transformer)                                | *Swin Transformer: Hierarchical Vision Transformer* (ICCV 2021)          | MIT                     |
| **ConvNeXt**                 | Image Classification | [Facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)                                          | *A ConvNet for the 2020s* (CVPR 2022)                                    | MIT                     |

# Object Detection Models

| **Model**                      | **Task**                   | **Code / Weights**                                                     | **Paper**                                                            | **License**           |
| ------------------------------ | -------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------- |
| **R-CNN**                      | Object Detection           | [Ross Girshick’s R-CNN (Caffe)](https://github.com/rbgirshick/rcnn)    | *Rich Feature Hierarchies for Accurate Object Detection* (CVPR 2014) | BSD 2-Clause          |
| **Fast R-CNN**                 | Object Detection           | [Fast R-CNN (Caffe)](https://github.com/rbgirshick/fast-rcnn)          | *Fast R-CNN* (ICCV 2015)                                             | MIT                   |
| **Faster R-CNN**               | Object Detection           | [Faster R-CNN (Python)](https://github.com/rbgirshick/py-faster-rcnn)  | *Faster R-CNN: Towards Real-Time Object Detection* (NIPS 2015)       | MIT                   |
| **YOLO (v1)**                  | Object Detection           | [Darknet (PJ Reddie)](https://github.com/pjreddie/darknet)             | *You Only Look Once: Unified Real-Time Detection* (CVPR 2016)        | YOLO License (custom) |
| **SSD** (Single Shot Detector) | Object Detection           | [Caffe SSD (Wei Liu)](https://github.com/weiliu89/caffe/tree/ssd)      | *SSD: Single Shot MultiBox Detector* (ECCV 2016)                     | BSD 2-Clause (Caffe)  |
| **RetinaNet**                  | Object Detection           | [Keras RetinaNet (Fizyr)](https://github.com/fizyr/keras-retinanet)    | *Focal Loss for Dense Object Detection* (ICCV 2017)                  | Apache 2.0            |
| **DETR**                       | Object Detection           | [Facebook DETR (PyTorch)](https://github.com/facebookresearch/detr)    | *End-to-End Object Detection with Transformers* (ECCV 2020)          | Apache 2.0            |
| **Mask R-CNN**                 | Object Det. + Segmentation | [Mask R-CNN (Matterport)](https://github.com/matterport/Mask_RCNN)     | *Mask R-CNN* (ICCV 2017)                                             | MIT                   |

# Segmentation Models

| **Model**                  | **Task**                   | **Code / Weights**                                                                        | **Paper**                                                                   | **License**             |
| -------------------------- | -------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------- |
| **FCN** (Fully Conv Net)   | Semantic Segmentation      | [FCN Reference (Caffe)](https://github.com/shelhamer/fcn.berkeleyvision.org)              | *Fully Conv. Networks for Semantic Segmentation* (CVPR 2015)                | BSD 2-Clause            |
| **U-Net**                  | Medical Image Segmentation | [U-Net Keras Implementation](https://github.com/zhixuhao/unet)                            | *U-Net: Conv Nets for Biomedical Segmentation* (MICCAI 2015)                | MIT                     |
| **DeepLab**                | Semantic Segmentation      | [DeepLab (TF Models)](https://github.com/tensorflow/models/tree/master/research/deeplab)  | *DeepLab: Semantic Image Segmentation with Atrous Convolution* (TPAMI 2017) | Apache 2.0              |
| **PSPNet**                 | Semantic Segmentation      | [PSPNet Official (Caffe)](https://github.com/hszhao/PSPNet)                               | *Pyramid Scene Parsing Network* (CVPR 2017)                                 | BSD 2-Clause (Caffe)    |
| **SegFormer**              | Semantic Segmentation      | [NVIDIA SegFormer (PyTorch)](https://github.com/NVlabs/SegFormer)                         | *SegFormer: Simple and Efficient Design for Segmentation* (NeurIPS 2021)    | NVIDIA Source Code (NC) |
| **SAM** (Segment Anything) | Promptable Segmentation    | [Meta Segment-Anything](https://github.com/facebookresearch/segment-anything)             | *Segment Anything* (CVPR 2023)                                              | Apache 2.0              |

# 3D & Video Models

| **Model**                         | **Task**                                   | **Code / Weights**                                               | **Paper**                                                       | **License**          |
| --------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- | --------------------------------------------------------------- | -------------------- |
| **PointNet**                      | 3D Point Cloud Classification/Segmentation | [PointNet (TensorFlow)](https://github.com/charlesq34/pointnet)  | *PointNet: Deep Learning on Point Sets* (CVPR 2017)             | MIT                  |
| **NeRF** (Neural Radiance Fields) | 3D View Synthesis (Novel View Gen.)        | [NeRF Official (TensorFlow)](https://github.com/bmild/nerf)      | *NeRF: Representing Scenes as Radiance Fields* (ECCV 2020)      | MIT                  |
| **3D CNNs** (e.g. C3D)            | Video Classification (Spatiotemporal CNN)  | [C3D (Caffe-based) Code](https://vlg.cs.dartmouth.edu/c3d/)      | *Learning Spatiotemporal Features with 3D ConvNets* (ICCV 2015) | BSD 2-Clause (Caffe) |
