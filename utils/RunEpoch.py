import torch
import os
from utils import Calculate, Log
import numpy as np


def train(opt, train_loader_set, val_loader_set, model, criterion, optimizer, scheduler, num_to_title):
    """
    模型的训练函数。
    @param opt: 初始化参数，由config.py定义
    @param train_loader_set: 训练集的加载器
    @param val_loader_set: 验证集的加载器
    @param model: 模型
    @param criterion: 损失计算函数
    @param optimizer: 优化器计算函数
    @param scheduler: 学习率优化函数
    @param num_to_title: 类别标签。dict{”index[类别标签]“：{”cn“:中文名字，“en”:英文名字}}
    """
    # -----------------------------------------------------
    # 1.初始化函数和参数
    # -----------------------------------------------------
    train_loss_calculate = Calculate.AverageMeter()
    log = Log.Log(opt, os.path.join(opt.save_path, "train_log.txt"), num_to_title)
    best = {'acc': 0, "best_epoch": 0}
    # -----------------------------------------------------
    # 2.开始训练
    # -----------------------------------------------------
    model.train()
    for i in range(opt.epoch):
        count_gt_label = np.zeros((opt.num_classes))
        train_acc = Calculate.Compute_acc(opt.num_classes)
        for index, data in enumerate(train_loader_set):
            img = data[0].to(opt.device)
            label = data[1].to(opt.device)
            # 记录每类的真实标签个数，方便后续计算每个标签的准确率
            for single_label in data[1]:
                count_gt_label[single_label] += 1
            pred = model(img)
            train_loss = criterion(pred, label.long())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step(epoch=i+1)
            # 计算准确率
            for batch in range(opt.batch_size):
                train_acc.update(pred[batch], label[batch])
            # 计算loss
            train_loss_calculate.update(train_loss.detach().cpu().numpy())
        pre_acc = train_acc.multi_class_calculate(opt, count_gt_label)

        log.write_json_log(pre_acc, train_loss_calculate.avg, 'train', i)
        log.write_log("TRAIN", i+1, train_loss_calculate.avg, train_acc.avg)
        log.write_csv("TRAIN", optimizer.state_dict()['param_groups'][0]['lr'], train_loss_calculate.avg, train_acc.avg)
        # -----------------------------------------------------
        # 3.开始验证
        # -----------------------------------------------------
        pre_val_acc, val_loss_calculate, val_acc = _validate(opt, val_loader_set, model, criterion, log, i, optimizer, num_to_title)

        # -----------------------------------------------------
        # 4.保存最优结果
        # -----------------------------------------------------
        if pre_val_acc["all"] > best['acc']:
            best['acc'] = pre_val_acc["all"]
            best['best_epoch'] = i
            torch.save(model.state_dict(), os.path.join(opt.save_path, "best.pth"))
        log.last_result(train_acc, val_acc, best)
        log.write_json_log(pre_val_acc, val_loss_calculate.avg, 'valid', i, best=best)
        log.create_json_log(i)

    torch.save(model.state_dict(), os.path.join(opt.save_path, "last.pth"))


def _validate(opt, val_loader_set, model, criterion, log, epoch, optimizer, num_to_title):
    """
    模型的验证函数.
    @param opt: 初始化参数，由config.py定义
    @param val_loader_set: 验证集的加载器
    @param model: 模型
    @param criterion: 损失计算函数
    @param epoch: 当前周期数
    @param num_to_title: 类别标签 dict{”index[类别标签]“：{”cn“:中文名字，“en”:英文名字}}
    @return:
        pre_val_acc->dict[包含每类的精确度]{"index":index_acc, "all": all_acc}
        val_loss_calculate->class[计算损失的实例,包含属性avg, acc]
    """
    # -----------------------------------------------------
    # 1.初始化函数和参数
    # -----------------------------------------------------
    val_loss_calculate = Calculate.AverageMeter()
    val_acc = Calculate.Compute_acc(opt.num_classes)
    count_gt_label = np.zeros((opt.num_classes))
    # -----------------------------------------------------
    # 2.开始验证
    # -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(val_loader_set):
            img = data[0].to(opt.device)
            label = data[1].to(opt.device)
            count_gt_label[label] += 1
            pred = model(img)
            val_loss = criterion(pred, label)
            val_loss_calculate.update(val_loss.detach().cpu().numpy())
            val_acc.update(pred, label)
            # -----------------------------------------------------
            # 3.展示模型输出结果,保存在output/picture下
            # -----------------------------------------------------
        log.write_log("VALID", epoch+1, val_loss_calculate.avg, val_acc.avg)
        log.write_csv("VALID", optimizer.state_dict()['param_groups'][0]['lr'], val_loss_calculate.avg, val_acc.avg)
        pre_val_acc = val_acc.multi_class_calculate(opt, count_gt_label)
    return pre_val_acc, val_loss_calculate, val_acc


