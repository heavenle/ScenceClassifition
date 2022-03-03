# -*- coding: UTF-8 -*-
__author__ = "liyi"
__data__ = "2022.2.21"
__description__ = "把数据格式变成场景分类需要的格式，原始格式为图片分类存储，目标格式如下," \
                  "#注意，改脚本只针对单标签！！"
"""
|——train
    |——images 
        |——1.jpg
        |——2.jpg 
        |——***.jpg
        |——N.jpg
    |——labels
        |——labels.csv
|——valid
    |——images
        |——3.jpg
        |——***.jpg
        |——M.jpg
    |——labels
        |——labels.csv
|——dataset.json
"""

import os
import json
import csv
import numpy as np
import glob
import shutil


def read_json(path):
    """
    读取提前写好的dataset.json
    :param path: dataset.json的位置
    :return: 读取好的字典。dict
    """
    with open(path, encoding='UTF-8') as F:
        conf = json.load(F)
    return conf


def to_classify(img_path, save_path):
    per_class_val_size = 40
    conf = read_json(os.path.join(save_path, "dataset.json"))
    # 在每类中随机选择20个作为验证集
    label_list = os.listdir(os.path.join(img_path))
    # 用来保存文件的初始参数
    val_count = 0
    train_count = 0
    train_f = open(os.path.join(save_path, "train", "labels", "labels.csv"), 'w', encoding='utf-8', newline="")
    val_f = open(os.path.join(save_path, "valid", "labels", "labels.csv"), "w", encoding='utf-8', newline="")
    train_csv_writer = csv.writer(train_f)
    val_csv_writer = csv.writer(val_f)
    train_csv_writer.writerow(["name", "cn_label"])
    val_csv_writer.writerow(["name", "cn_label"])
    for label in label_list:
        img_list = glob.glob(os.path.join(img_path, label, "*.tif"))
        per_class_all_size = len(img_list)
        # per_class_all_size = 200
        random_index = np.random.choice([x for x in range(per_class_all_size)], per_class_val_size, replace=False)
        for i in range(per_class_all_size):
            if i in random_index:
                val_count += 1
                img = img_list[i]
                val_csv_writer.writerow([str(val_count) + ".tif", conf[label]["title"]])
                shutil.copy(img, os.path.join(save_path, "valid", "images", str(val_count) + ".tif"))
            else:
                train_count += 1
                img = img_list[i]
                train_csv_writer.writerow([str(train_count) + ".tif", conf[label]["title"]])
                shutil.copy(img, os.path.join(save_path, "train", "images", str(train_count) + ".tif"))
        print("label:{}==>save val count[{}]|save train count[{}]".format(label, val_count, train_count))
    train_f.close()
    val_f.close()

if __name__ == '__main__':
    img_path = r"F:\database\scene_classifition\Google_dataset_of_SIRIWHU_earth_im_tiff\Images"
    save_path = r"F:\database\scene_classifition\Google_dataset_of_SIRIWHU_earth_im_tiff\pie"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    to_classify(img_path, save_path)