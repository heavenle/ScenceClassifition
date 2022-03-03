# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for training model"

import os
import torch
import torch.utils.data.dataloader as dataloader
import torchvision.transforms.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR
from config import Config
from dataset import create_dataset
from model import create_model
from loss import create_loss
import json
from utils import RunEpoch


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def main():
    # -----------------------------------------------------
    # 1.初始化参数
    # -----------------------------------------------------
    config = Config()
    opt = config.init_config()
    for key, value in vars(opt).items():
        print('===>['+ key + ':' + str(value)+']')
    # -----------------------------------------------------
    # 2.创建工作路径
    # -----------------------------------------------------
    if os.path.exists(opt.save_path):
        print("currently file name is repeat, suggesting rename file name[\"work_dirs/[new_name]\"]")
        print("[NOTE]you work_dirs/[{}] will remove all files.".format(os.path.basename(opt.save_path)))
        enter = input('please enter y to confirm: ')
        if enter == 'y':
            del_file(opt.save_path)
        else:
            print("plese enter correct content")
            exit(0)
    else:
        os.mkdir(opt.save_path)
    # -----------------------------------------------------
    # 3.创建数据集合
    # -----------------------------------------------------
    cn_index = dict()
    num_to_title = dict()
    with open(os.path.join(opt.data, "dataset.json"), encoding='UTF-8') as F:
        conf = json.load(F)
    for key, _ in conf.items():
        cn_index[conf[key]["title"]] = conf[key]["index"]
        num_to_title[conf[key]["index"]] = {'cn': conf[key]["title"],
                                            'en': key}
    train_transform = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor()])

    train_data_set = create_dataset(opt.dataset)(os.path.join(opt.data, "train", "images"),
                                                 os.path.join(opt.data, "train", "labels", "labels.csv"),
                                                 cn_index,
                                                 opt,
                                                 train_transform)
    train_loader_set = dataloader.DataLoader(train_data_set,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=0)

    val_transform = transforms.Compose([transforms.ToTensor()])
    val_data_set = create_dataset(opt.dataset)(os.path.join(opt.data, "valid", "images"),
                                               os.path.join(opt.data, "valid", "labels", "labels.csv"),
                                               cn_index,
                                               opt,
                                               val_transform)
    val_loader_set = dataloader.DataLoader(val_data_set,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0)
    print("==================>Finished data loading==================>")
    # -----------------------------------------------------
    # 4.创建模型。
    # -----------------------------------------------------
    model = create_model(opt.arch)(opt, pretrained=opt.pretrained)
    print("==================>Finished model loading==================>")
    # -----------------------------------------------------
    # 5.设置网络超参数
    # -----------------------------------------------------
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=0.00001)

    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 200, 250], gamma=0.1)
    criterion = create_loss(opt.loss)
    # reg_loss = create_loss('Regularization', model, weight_decay=1, p=2)
    # -----------------------------------------------------
    # 6.设置GPU,目前只支持单GPU
    # -----------------------------------------------------
    if len(opt.gpu) == 1:
        opt.device = "cuda:" + str(opt.gpu[0])
    else:
        print("this function is not complete!")
        exit(0)
    model = model.to(opt.device)
    # -----------------------------------------------------
    # 7.开始训练。
    # -----------------------------------------------------
    print("=================>start training==================>")
    RunEpoch.train(opt, train_loader_set, val_loader_set, model, criterion, optimizer, scheduler, num_to_title)


if __name__ == "__main__":
    main()