import glob
import random
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageEnhance

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class VOSDataset(Dataset):
    """
    image suffix: jpg
    mask suffix: png
    """
    def __init__(self, cfg, is_train, datasets):

        self.train_support_dataset = ["YouTubeVOS-2018", "DAVIS-2016"]
        self.val_support_dataset = ["DAVIS-2016", "FBMS", "ViSal"]

        img_size = cfg.img_size
        mean = cfg.mean
        std = cfg.std

        self.datasets = datasets
        self.data_dir = cfg.data_dir
        self.stride = cfg.stride
        self.is_train = is_train
        self.images, self.masks = [], []

        # load dataset
        self.split_dataset(is_train=is_train)
        # transform
        size = self.get_size(img_size=img_size)
        self.image_transform = self.get_image_transform(size=size, mean=mean, std=std)
        self.mask_transform = self.get_flow_mask_transform(size=size)

        assert len(self.images) == len(self.masks)

    def split_dataset(self, is_train):
        if is_train:
            support_dataset = self.train_support_dataset
        else:
            support_dataset = self.val_support_dataset

        for dataset in self.datasets:
            if dataset in support_dataset:
                self.load_dataset(dataset=dataset, is_train=is_train)
            else:
                raise ValueError(f"Not support this dataset: {dataset}")

    def load_dataset(self, dataset, is_train):

        if is_train:
            data_dir = osp.join(self.data_dir, dataset, "train")
        else:
            data_dir = osp.join(self.data_dir, dataset, "val")

        assert os.listdir(osp.join(data_dir, "images")) == os.listdir(osp.join(data_dir, "labels")), \
            "video number or video name are different between images and labels."

        videos = os.listdir(osp.join(data_dir, "images"))
        for video in videos:
            # image
            frames = sorted(glob.glob(osp.join(data_dir, "images", video, "*.jpg")))
            image_frames = []
            for i in range(len(frames) - 1):
                image_frames.append([frames[i], frames[i + 1]])

            # mask
            mask_frames = sorted(glob.glob(osp.join(data_dir, "labels", video, "*.png")))
            mask_frames.pop(0)

            assert len(image_frames) == len(mask_frames)

            if is_train:
                # only sample YouTubeVOS
                if self.stride > 1 and dataset == "YouTubeVOS-2018":
                    image_frames = image_frames[::self.stride]
                    mask_frames = mask_frames[::self.stride]

            for i in range(len(image_frames)):
                self.images.append(image_frames[i])
                self.masks.append(mask_frames[i])

    def get_size(self, img_size):
        if isinstance(img_size, int):
            size = (img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
            size = (img_size[0], img_size[0])
        else:
            assert len(img_size) == 2, f"image size: {img_size} > 2 and is not a image"
            size = img_size
        return size

    def get_image_transform(self, size, mean, std):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform

    def get_flow_mask_transform(self, size):
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        return transform

    def __getitem__(self, idx):

        image_pair, mask = self.images[idx], self.masks[idx]
        image1, image2 = image_pair[0], image_pair[1]
        image1 = Image.open(image1).convert("RGB")
        image2 = Image.open(image2).convert("RGB")
        mask = Image.open(mask).convert("P")
        o_shape = image1.size
        mask_path = self.masks[idx]
        image1, image2, mask = self.data_augmentation(image1, image2, mask)

        return image1, image2, mask, o_shape, mask_path

    def data_augmentation(self, image1, image2, mask):
        if self.is_train:
            image1, image2, mask = self.random_crop(image1, image2, mask, border=60)
            image1, image2, mask = self.cv_random_flip(image1, image2, mask)
            image1,image2, mask = self.random_rotation(image1, image2, mask)
            image1 = self.color_enhance(image1)
            image2 = self.color_enhance(image2)
        image1 = self.image_transform(image1)
        image2 = self.image_transform(image2)
        mask = self.mask_transform(mask)

        return image1, image2, mask

    def random_rotation(self, image1, image2, mask):
        mode = Image.Resampling.BICUBIC
        if random.random() > 0.5:
            random_angle = np.random.randint(-10, 10)
            image1 = image1.rotate(random_angle, mode)
            image2 = image2.rotate(random_angle, mode)
            mask = mask.rotate(random_angle, mode)
        return image1, image2, mask

    def cv_random_flip(self, image1, image2, mask):
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            image1 = image1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image1, image2, mask

    def color_enhance(self, image):
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return image

    def random_crop(self, image1, image2, mask, border=30):
        image_width = mask.size[0]
        image_height = mask.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
            (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return image1.crop(random_region), image2.crop(random_region), mask.crop(random_region)

    def __len__(self):
        return len(self.masks)