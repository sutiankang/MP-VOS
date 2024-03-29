# 基于并行多方向注意力的无监督视频目标分割(MP-VOS)

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.8.1 with two GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.8
```
conda create -n mp-vos python=3.8
```

Other minor Python modules can be installed by running
```
pip install -r requirements.txt
```

## Train

### Download Datasets
In the paper, we use the following three public available dataset for training. Here are some steps to prepare the data:

- [DAVIS-2016](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 9 frames in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter ```--stride```.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We use all the data in the train subset of FBMS.

The structure of datasets is as follows:
```
|—— Datasets
  |—— YouTubeVOS-2018
    |—— train
      |—— images
        |—— 00000.jpg
        |—— 00005.jpg
        |—— ...
      |—— labels
        |—— 00000.png
        |—— 00005.png
        |—— ...
    |—— val
      |—— images
      |—— labels    
  |—— DAVIS-2016
    |—— train
      |—— images
      |—— labels    
    |—— val
      |—— images
      |—— labels    
  |—— FBMS
    |—— train
      |—— images
      |—— labels    
    |—— val
      |—— images
      |—— labels    
```

### Train
- First, train the model using the YouTubeVOS-2018.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --sync_bn --pretrained
```
- Second, finetune the model using the DAVIS-2016.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --sync_bn --finetune first_stage_weight_path
```

## Test

-   Run following to generate the segmentation results.
```
python test.py --weights model_weight_path --save_dir save_results_path --use_flip -v test_datasets
```
