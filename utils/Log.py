# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "The script file is used for recording model output text"

import time
import os
import csv
import json

class Log:
    def __init__(self, opt, log_save_path, num_to_title):
        """
        log的初始函数
        @param opt:初始化参数,由config.py定义
        @param log_save_path: log的保存路径
        @param num_to_title: 类别标签 dict{”index[类别标签]“：{”cn“:中文名字，“en”:英文名字}}
        """
        self.log_save_path = log_save_path
        self.opt = opt
        with open(self.log_save_path, 'a') as F:
            F.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for key, value in vars(self.opt).items():
                F.write(key + ':' + str(value))
                F.write('\n')
            F.write('\n')

        with open(os.path.join(os.path.dirname(self.log_save_path), "log_train.csv"),
                 "a", encoding='utf-8', newline="") as train_f:
            self.train_csv_writer = csv.writer(train_f)
            self.train_csv_writer.writerow(["lr", "loss", "acc"])

        with open(os.path.join(os.path.dirname(self.log_save_path), "log_val.csv"),
                 "a", encoding='utf-8', newline="") as val_f:
            self.val_csv_writer = csv.writer(val_f)
            self.val_csv_writer.writerow(["lr", "loss", "acc"])

        self.num_to_title = num_to_title
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.epoch_log = {"epoch": 1, "train": [], "valid": []}
        self.log = {
            'training_id': "",
            "state": "",
            "create_time": start_time,
            "description": {
                "acc": "准确率",
                "loss": "损失值"
            },
            "epochs": [],
            'best': "",
            "end_time": ""
        }

    def write_log(self, form, epoch, loss, acc):
        """
        写入log
        @param form:状态,判断是TRAIN和VALID
        @param epoch:当前周期数
        @param loss:当前输入损失
        @param acc:当前输入精确度
        """
        if form == 'VALID' or 'TRAIN':
            with open(self.log_save_path, "a") as F:
                F.write("[{}]--epoch[{}/{}]--loss:{}--acc{}".format(form, epoch, self.opt.epoch,
                                                                                 loss, acc))
                F.write('\n')
            print("[{}]--epoch[{}/{}]--loss:{}--acc{}".format(form, epoch, self.opt.epoch,
                                                                            loss, acc))
        else:
            print("please write true form[VALID or TRAIN]")
            exit(0)

    def write_csv(self, form, lr, loss, acc):
        """
        写入CSV文件
        @param form: 状态,判断是TRAIN和VALID
        @param lr: 当前学习率
        @param loss: 当前输入loss
        @param acc: 当前输入准确率
        """
        if form == 'VALID':
            with open(os.path.join(os.path.dirname(self.log_save_path), "log_val.csv"),
                      "a", encoding='utf-8', newline="") as val_f:
                self.val_csv_writer = csv.writer(val_f)
                self.val_csv_writer.writerow([str(lr), str(loss), str(acc)])
        elif form == 'TRAIN':
            with open(os.path.join(os.path.dirname(self.log_save_path), "log_train.csv"),
                      "a", encoding='utf-8', newline="") as train_f:
                self.train_csv_writer = csv.writer(train_f)
                self.train_csv_writer.writerow([str(lr), str(loss), str(acc)])
        else:
            print("please write true form[VALID or TRAIN]")
            exit(0)

    def last_result(self, train_acc, valid_acc, best):
        """
        将最后的准确率结果保存的log中
        @param train_acc: 计算训练集准确的类实例, 包含属性 acc 和avg
        @param valid_acc: 计算验证集准确的类实例, 包含属性 acc 和avg
        @param best: 最优的准确率和所在周期. dict{"best_acc":acc, "best_epoch":epoch}
        """
        if train_acc == None or valid_acc == None:
            print("train_acc and val_acc is None")
            exit(0)
        with open(self.log_save_path, "a") as F:
            F.write("-" * 30)
            F.write('\n')
            F.write("train_acc:{}".format(train_acc.avg))
            F.write('\n')
            F.write("train_pred_label:{}[acc of per class]".format(train_acc.acc))
            F.write('\n')
            F.write("valid_acc:{}".format(valid_acc.avg))
            F.write('\n')
            F.write("valid_pred_label:{}[acc of per class]".format(valid_acc.acc))
            F.write('\n')
            F.write("best_acc:{}".format(best['acc']))
            F.write('\n')
            F.write("best_epoch:{}".format(best['best_epoch']))
            F.write('\n')
            F.write("-" * 30)
            F.write('\n')
        print("-" * 30)
        print("train_acc:{}".format(train_acc.avg))
        print("train_pred_label:{}[acc of per class]".format(train_acc.acc))
        print("valid_acc:{}".format(valid_acc.avg))
        print("valid_pred_label:{}[acc of per class]".format(valid_acc.acc))
        print("-" * 30)


    def write_json_log(self, acc, loss, status, epoch, best=None):
        """
        记录每个周期的输出结果，并在验证时获取最优参数。
        params: acc 准确率 -> float
        params: loss 所有类别的平均损失值 -> float

        """
        self.epoch_log[status].append({
            "class": "all",
            "values": {
                "acc": acc["all"],
                "loss": loss
            }
        })
        for key, name in self.num_to_title.items():
            self.epoch_log[status].append({
                "class": name['en'],
                "values": {
                    "acc": acc[str(key)],
                }
            })

        if status == 'valid':
            self.best = {
                "acc": best['acc'],
                "epoch": best['best_epoch']+1,
            }

    def create_json_log(self, epoch):
        """
        将每个周期后的结果写入trainlog.json中。
        该函数应放在每个周期的结尾。
        params:epoch 当前周期
        """
        if self.log is None:
            print('currently class did not define self.log, please use creat_log function!')
            exit(0)

        self.log["epochs"].append(self.epoch_log)
        self.epoch_log = {"epoch": str(epoch + 1), "train": [], "valid": []}
        self.log["best"] = self.best
        self.opt.state = 0
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log["end_time"] = end_time

        trainlog = json.dumps(self.log, indent=4)
        with open(os.path.join(os.path.dirname(self.log_save_path), 'trainlog' + '.json'), 'w',
                  encoding='utf-8') as log_file:
            log_file.write(trainlog)

