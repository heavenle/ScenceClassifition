# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "linfenliyi@gmail.com"
__data__ = "2022.3.1"
__description__ = "The script file is used for predict big image"

import os
import torch
from config import Config
from model import vgg, resnet
import json
import torchvision.transforms.transforms as transforms
from dataset import create_dataset
import torch.utils.data.dataloader as dataloader
from utils import Calculate, Log


def val():
    # -----------------------------------------------------
    # 1.初始化参数
    # -----------------------------------------------------
    config = Config()
    opt = config.init_config()
    print(opt)
    # -----------------------------------------------------
    # 2.创建数据集合
    # -----------------------------------------------------
    cn_index = dict()
    num_index = dict()
    num_to_color = dict()
    with open(os.path.join(opt.data, "dataset.json"), encoding='UTF-8') as F:
        conf = json.load(F)
    for key, _ in conf.items():
        cn_index[conf[key]["title"]] = conf[key]["index"]
        num_index[conf[key]["index"]] = conf[key]["title"]
        num_to_color[conf[key]["index"]] = conf[key]["color"]

    transform = transforms.Compose([transforms.ToTensor()])

    val_data_set = create_dataset(opt.dataset)(os.path.join(opt.data, "valid", "images"),
                                               os.path.join(opt.data, "valid", "labels", "labels.csv"),
                                               cn_index,
                                               opt,
                                               transform)
    val_loader_set = dataloader.DataLoader(val_data_set,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0)

    print("=============>Finished data loading")
    # -----------------------------------------------------
    # 3.创建模型。
    # -----------------------------------------------------
    model = resnet.resnet101(opt, pretrained=False)
    state_dict = torch.load("F:\\scence_classifition\\work_dirs\\UCM_resnet101\\best.pth")
    model.load_state_dict(state_dict)
    print("=============>Finished model loading")
    # -----------------------------------------------------
    # 4.设置网络超参数
    # -----------------------------------------------------
    if len(opt.gpu) == 1:
        opt.device = "cuda:" + str(opt.gpu[0])
    else:
        print("this function is not complete!")
        exit(0)
    model = model.to(opt.device)
    # -----------------------------------------------------
    # 5.开始测试。
    # -----------------------------------------------------
    print("=============>start training")
    val_acc = Calculate.Compute_acc(opt.num_classes)
    for index, data in enumerate(val_loader_set):
        model.eval()
        with torch.no_grad():
            img = data[0].to(opt.device)
            label = data[1].to(opt.device)
            pred = model(img)
            val_acc.update(pred, label)
    print(val_acc.avg)

if __name__ == "__main__":
    val()