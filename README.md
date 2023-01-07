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
