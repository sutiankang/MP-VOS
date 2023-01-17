# 基于并行多方向注意力的无监督视频目标分割(MP-VOS)

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.8.1 with two GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.6
```
conda create -n mp-vos python=3.6
```

## Train

### Download Datasets
In the paper, we use the following three public available dataset for training. Here are some steps to prepare the data:

- [DAVIS-16](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 5 frames in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter ```--stride```.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We use all the data in the train subset of FBMS.

The structure of datasets is as follows:
```
|—— Datasets
  |—— YouTubeVOS2018
    |—— train
    |—— val
  |—— DAVIS-2016
    |—— train
    |—— val
  |—— FBMS
    |—— train
    |—— val
```
