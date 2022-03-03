# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for calculating model metrics "


import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        初始化
        @param num_classes: 类别数
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        计算总loss
        @param val: 输入的loss值
        @param n: batch_size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Compute_acc(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_classes):
        self.reset(num_classes)

    def reset(self, num_classes):
        """
        初始化
        @param num_classes: 类别数
        """
        self.avg = 0
        self.count = 0
        self.acc = np.zeros((num_classes))

    def update(self, pred, label, n=1):
        """
        计算总精确率
        @param pred:模型预测结果 tensor[batch, num_classes]
        @param label: 真实标签 tensor[batch]
        @param n: batch_size
        """
        pred = pred.detach().cpu().numpy().argmax()
        if label == pred:
            self.acc[label] += 1
        self.count += n
        self.avg = self.acc.sum()/ self.count

    def multi_class_calculate(self, opt, count_gt_label):
        """
        计算多类的准确率
        @param opt:初始化参数,由config.py定义
        @param count_gt_label:每类的真实标签的数量.numpy[num_classes]
        @return:每类的精确率 dict{"index":index_acc,"all":all_acc}
        """
        pre_acc = {str(x): 0 for x in range(opt.num_classes)}
        pre_acc["all"] = 0
        for index, value in enumerate(count_gt_label):
            pre_acc[str(index)] = self.acc[index] / value
        pre_acc["all"] = self.avg
        return pre_acc


