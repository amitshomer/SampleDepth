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

### SHIFT
Download the data by using [SHIFT Repo](https://github.com/SysCV/shift-dev) and the script within it. 

The structure of the dataset should be as follows:
```
|--discrete
  |-- images
    |--test
    |--train
    |--val
      |--scene
        |--xxxx_depth_front.png, xxxx_img_front.png
```
## Evaluate SampleDepth
You can find and [download our pre-trained weights](https://drive.google.com/drive/u/0/folders/1ka-7uHzW6x2S9d67NXaZL3cnQCvdqN9p?hl=he) for the SHIFT dataset for the **End-to-End** and **Lower-bound** versions.

The following command can be used to evaluate our **End-to-End based PredNet*** sampler on the SHIFT dataset:
```
Python End_to_End.py –batch_size 1 –nworkers 8 –dataset SHIFT –multi True –sampler_input gt –sampler_type SampleDepth –n_sample 19000 –gpu_device 0 –evalute –eval_path_random_model {Path to EndtoEnd_random_task} –eval_path_predNet {Path to EndtoEnd_PredNet.pth} –eval_path_SampleDepth {Path to EndtoEnd_SampleDepth.pth}
```

The following command can be used to evaluate our **Lower-bound*** sampler on the SHIFT dataset:
```
Python main_sampler.py –batch_size 1 –nworkers 8 –dataset SHIFT –multi True –sampler_input gt –sampler_type SampleDepth –n_sample 19000 –evaluate –eval_path {Path to  SHIFT_LowerBound.pth.tar}
```

## Train SampleDepth

### Lower bound training flow
1. Train task network:
```
Python main_task.py –batch_size 20 –dataset SHIFT –nworkers 8 –multi True -n_sample 19000 –sampler_input gt –sample_factor_type 'n_points' –learning_rate 0.008
```

2. Train SampleDepth: 
```
python main_sampler.py --–batch_size 10 –dataset SHIFT –nworkers 8 –multi True -n_sample 19000 –sampler_input gt --learning_rate 0.0001 –alpha 0.1 –beta 20 --task_weight {path_weights_from_stage1}
```

3. Finetune SampleDepth+Task network:
```
python main_sampler.py --–batch_size 10 –dataset SHIFT –nworkers 8 –multi True -n_sample 19000 –sampler_input gt --learning_rate 0.0001 –alpha 0.1 –beta 10--fine_tune --finetune_path {path_weights_from_stage2 }
```

### Fixed mask training flow
```
python main_sampler.py --batch_size 1 --dataset SHIFT –nworkers 8 –multi True -n_sample 19000 –sampler_input gt –learning_rate 0.001 --sampler_type global_mask --alpha 1 –beta 20 --task_weight {Path to Lower-bound stage 1}
```


### Adaptive PredNet
1. Ensure that you have the Lower-bound weights from stage 3. They can also be found [here](https://drive.google.com/drive/u/0/folders/1ka-7uHzW6x2S9d67NXaZL3cnQCvdqN9p?hl=he). 

2. Prior to stage 2, a preprocessing step is required on the dataset. Please refer to comment 1.

3. Train PredNet(Only a prediction of the current frame based on past frames):
```
python main_predict_depth.py --batch_size 10 --dataset SHIFT –nworkers 8 --multi True --model_type Unet --learning_rate 0.0001 --past_inputs 4 --reconstructed_folder {see_comment1_below}
```

4. Prior to stage 3, a preprocessing step is required on the dataset. Please refer to comment 2.

5. Train SampleDepth with PredNet predication prior:
```
python main_sampler.py --batch_size 10 --dataset SHIFT --nworkers 8 --multi True --n_sample 19000 --sampler_input gt --learning_rate 0.0001 --alpha 0.1 --beta 20 --sampler_input predict_from_past --past_inputs 4 --fine_tune --finetune_path {path from Lower-bound weights}
```

