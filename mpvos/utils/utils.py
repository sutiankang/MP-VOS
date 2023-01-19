import torch
import numpy as np


def get_size(img_size):
    if isinstance(img_size, int):
        size = (img_size, img_size)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        size = (img_size[0], img_size[0])
    else:
        assert len(img_size) == 2, f"image size: {img_size} > 2 and is not a image"
        size = img_size
    return size


def create_dataset(args, datasets, is_train):
    if isinstance(datasets, str):
        datasets_list = [datasets]
    elif isinstance(datasets, list):
        datasets_list = datasets
    else:
        raise ValueError

    from mpvos.datasets import VOSDataset
    vos_dataset = VOSDataset(args, is_train=is_train, datasets=datasets_list)

    return vos_dataset


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)