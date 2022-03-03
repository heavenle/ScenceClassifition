# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for predicting big image"

import os
import torch
from config import Config
from model import create_model
import json
import glob
from utils import predict_utils
from osgeo import gdal


def predict():
    # -----------------------------------------------------
    # 1.初始化参数
    # -----------------------------------------------------
    config = Config()
    opt = config.init_config()
    print(opt)

    if not os.path.exists(os.path.join(opt.save_path, 'picture')):
        os.mkdir(os.path.join(opt.save_path, 'picture'))
    # -----------------------------------------------------
    # 2.获取数据列表
    # -----------------------------------------------------
    img_list = glob.glob(os.path.join(opt.predict_data, "*"))

    cn_index = dict()
    num_index = dict()
    num_to_color = dict()
    with open(os.path.join(opt.data, "dataset.json"), encoding='UTF-8') as F:
        conf = json.load(F)
    for key, _ in conf.items():
        cn_index[conf[key]["title"]] = conf[key]["index"]
        num_index[conf[key]["index"]] = conf[key]["title"]
        num_to_color[conf[key]["index"]] = conf[key]["color"]

    pie = predict_utils.PredictUtils(opt, num_to_color)
    print("=============>Finished data loading")
    # -----------------------------------------------------
    # 3.创建模型。
    # -----------------------------------------------------
    model = create_model(opt.arch)(opt, pretrained=False)
    state_dict = torch.load(os.path.join(opt.save_path, 'best.pth'))
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
    for index, img_path in enumerate(img_list):
        im_data, dataset = pie.read_geo_img(img_path, gdal)
        pie.pred_big_geo_img(index, opt, model, im_data, dataset, img_path,
                             slide_window_size=opt.imgsize, overlap_rate=0)


if __name__ == "__main__":
    predict()