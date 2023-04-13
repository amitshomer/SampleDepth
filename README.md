# Prior based Sampling for Adaptive LiDAR (SampleDepth)
This repo contains the implementation of our paper [Prior based Sampling for Adaptive LiDAR]() by Amit Shomer and Shai Avidan from Tel Aviv University.

<img src="https://github.com/amitshomer/SampleDepth/blob/master/docs/teaser_new.png" width=50% height=50%>

## Citation
If you find our work useful in your research, please consider citing: 
```
add citing
```
## Introduction
We propose SampleDepth, a Convolutional Neural Network (CNN), that is suited for an adaptive LiDAR. 

Typically, LiDAR sampling strategy is pre-defined, con-stant and independent of the observed scene. Instead of letting a LiDAR sample the scene in this agnostic fashion, SampleDepth determines, adaptively, where it is best to sample the current frame. To do that, SampleDepth uses depth samples from previous time steps to predict a sampling mask for the current frame. 

Crucially, SampleDepth is trained to optimize the performance of a depth completion downstream task. SampleDepth is evaluated on two different depth completion networks and two LiDAR datasets, KITTI Depth Completion and the newly introduced synthetic dataset, SHIFT. We show that SampleDepth is effective and suitable for different depth completion downstream tasks.

## Requirements
Pytorch 1.9, CUDA 11.4, Ubuntu 18.04.5. 

Install the environment using the yml file:
`conda env create -f environment.yml`

## Dataset
### KITTI Depth Completion
First, download the [Depth Completion](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset. Next, download the RGB images from the [KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php). 
Additionally, we provide a script that may be used to download the dataset: `Download/download_raw_files.sh`

The structure of the dataset should be as follows:
```
|--depth selection
|-- train
  |--date
    |--sequence_number
      |--image_02
        |--data
          |--proj_depth
          |img1.png, img2.png…
|--val
```

