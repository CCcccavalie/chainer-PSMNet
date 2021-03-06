import os
import random
from PIL import Image, ImageOps
from .readpfm import readPFM 
import numpy as np
import cv2

import chainer
from chainercv import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return readPFM(path)


class myImageFolder(chainer.dataset.DatasetMixin):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __len__(self):
        return len(self.left)

    def get_example(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img, right_img = custom_transform(left_img, right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 960, h - 540, w, h))
            right_img = right_img.crop((w - 960, h - 540, w, h))
            left_img, right_img = custom_transform(left_img, right_img)

            return left_img, right_img, dataL


def custom_transform(left_img, right_img, augment=False):
    """
    if augment:
        # 1. RandomSizedCrop
        #left_img, param = transforms.random_sized_crop(left_img, return_param=True)
        # 2. RandomHorizontalFlip
        #img = transforms.random_flip(img, x_random=True)
        # 3. ColorJitter
        # img =
        # 4. Lighting
        left_img = transform.pca_lighting(
            left_img, 0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']
        )
        right_img = transform.pca_lighting(
            right_img, 0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']
        )
    """

    # temporal normal preprocess
    w, h = left_img.size

    #print(left_img.size)
    #cv2.imwrite("left.png",np.asarray(left_img))
    #cv2.imwrite("right.png",np.asarray(right_img))
    #exit()

    # reshape into tensor
    left_img = np.asarray(left_img).transpose(2,0,1).reshape(3, h, w)
    right_img = np.asarray(right_img).transpose(2,0,1).reshape(3, h, w)

    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)
    left_img /= 255.
    right_img /= 255.

    return left_img, right_img
