# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for saving dataloader of Google_Dataset_of_SIRIWHU dataset "

from torch.utils.data import Dataset
import skimage.io as io
import glob
import os
import pandas as pd
import numpy as np
import cv2

class Google_Dataset_of_SIRIWHU(Dataset):
    def __init__(self, img_path, label_path, cn_index, opt, transformer=None):
        self.img_path = glob.glob(os.path.join(img_path, "*.jpg"))
        self.lable_path = label_path
        self.csv_vaule = np.array(pd.read_csv(self.lable_path))
        self.T = transformer
        self.cn_index = cn_index
        self.opt = opt

    def __getitem__(self, item):
        images = io.imread(self.img_path[item])
        labels = None
        for index in range(self.csv_vaule.shape[0]):
            if self.csv_vaule[index][0] == os.path.basename(self.img_path[item]):
                labels = self.cn_index[self.csv_vaule[index][1]]
        if images.shape[0] != 200 or images.shape[1] != 200:
            images = cv2.resize(images, (200, 200))
        images = images.astype('float32') / 255
        if self.T:
            images = self.T(images)
        return images, labels

    def __len__(self):
        return len(self.img_path)
