# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for defining initialization parameters"


import argparse


class Config:
    def __init__(self, config_path=None):
        super(Config, self).__init__()
        self.opt = None
        self.path = config_path

    def init_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data",
                            default=r'F:\database\scene_classifition\Google_dataset_of_SIRIWHU_earth_im_tiff\pie',
                            help="the dataset path.")
        parser.add_argument("--predict_data",
                            default=r'F:\database\scene_classifition\test',
                            help="the predicted dataset path.")
        parser.add_argument("--save_path",
                            default=r'./work_dirs/SIRIWHU_resnet101',
                            help="save file path.")

        parser.add_argument("--dataset",
                            default="General_template",
                            help="dataset name ['General_template',"
                                                "'Google_dataset_of_SIRIWHU_earth_im_tiff',"
                                                "'UCMerced_LandUse',"
                                                "'CatVsDog']")
        parser.add_argument("--imgsize",
                            default=200,
                            help="resize image [int]")
        parser.add_argument("--loss",
                            default="cross_entropy", help="loss name ['cross_entropy']")
        parser.add_argument("--lr",
                            default=0.0001, help="learning rate")
        parser.add_argument("--batch_size",
                            default=8, help="batch size")
        parser.add_argument("--gpu",
                            default=[0], help="gpu name, no support multi-gpu")
        parser.add_argument("--epoch",
                            default=2, help="the size of epoch")
        parser.add_argument("--num_classes",
                            default=12, help="num_classes")
        parser.add_argument("--arch",
                            default='resnet18', help="arch[vgg19_bn, vgg19, resnet18, resnet101, mobilenetv2]")
        parser.add_argument("--pretrained", default=True,
                            help="Whether ImageNet pre-training weights are required")
        self.opt = parser.parse_args()
        return self.opt
