
"""
train segmentation
"""

import os
import sys
# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# 将models文件夹的路径添加到sys.path中，使得models文件夹中的py文件能被本文件import
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

# 工具包
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import logging # 记录日志信息
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init

# 自建模块
from data_utils.ParamDataLoader import Seg360GalleryDataLoader
from data_utils.ParamDataLoader import segfig_save
from models.cstnet_seg import CrossAttention_Seg
from models.cst_pred import TriFeaPred_OrigValid


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # 指定的GPU设备
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--epoch', default=1, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training') # 优化器
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')

    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量

    parser.add_argument('--save_str', type=str, default='ca_final_predattr_part_seg', help='---')
    parser.add_argument('--is_show_img', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--is_use_pred_addattr', type=str, default='True', choices=['True', 'False'], help='---')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\360Gallery_Seg', help='root of dataset')

    # 参数化数据集：D:/document/DeepLearning/DataSet/data_set_p2500_n10000
    # 参数化数据集(新迪)：r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud'
    # 参数化数据集(新迪，服务器)：r'/opt/data/private/data_set/PointCloud_Xindi_V2/'
    # modelnet40数据集：r'D:\document\DeepLearning\DataSet\modelnet40_normal_resampled'
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_B
    # 360 Gallery: D:\document\DeepLearning\DataSet\360Gallery_Seg

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def compute_instance_miou(seg_pred, target, n_seg_part):
    """
    计算单个 batch 的 Instance MIOU
    Args:
        seg_pred: torch.Tensor, 大小为 [batch_size, n_points, n_seg_part]
        target: torch.Tensor, 大小为 [batch_size, n_points]
        n_seg_part: int, 分割的部分数量
    Returns:
        Instance MIOU: float
    """
    batch_size = seg_pred.shape[0]

    # 1. 获取每个点的预测分割部分
    seg_pred_label = torch.argmax(seg_pred, dim=-1)  # [batch_size, n_points]

    iou_list = []

    # 2. 对每个 batch 中的实例分别计算 IOU
    for b in range(batch_size):
        iou_per_part = []
        for part in range(n_seg_part):
            # 计算预测和标签中属于当前 part 的点
            pred_mask = (seg_pred_label[b] == part)  # 预测属于该 part 的点
            target_mask = (target[b] == part)  # 真实属于该 part 的点

            # 计算交集（TP）和并集（TP + FP + FN）
            intersection = torch.sum(pred_mask & target_mask).item()
            union = torch.sum(pred_mask | target_mask).item()

            if union == 0:
                # 如果并集为 0，说明该 part 在预测和标签中都不存在，跳过
                continue
            else:
                iou = intersection / union  # 计算 IOU
                iou_per_part.append(iou)

        if iou_per_part:
            # 计算该实例的平均 IOU
            instance_miou = sum(iou_per_part) / len(iou_per_part)
            iou_list.append(instance_miou)

    if iou_list:
        # 计算整个 batch 的平均 IOU
        return sum(iou_list) / len(iou_list)
    else:
        return 0.0


def clear_log(log_dir):
    """
    清空空白的log文件
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(log_dir):
        # 获取文件的完整路径
        file_path = os.path.join(log_dir, filename)
        # 检查是否为txt文件且为空
        if filename.endswith('.txt') and os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    if args.is_show_img == 'True':
        is_show_img = True
    else:
        is_show_img = False

    if args.is_use_pred_addattr == 'True':
        is_use_pred_addattr = True
        print(Fore.GREEN + 'use predict parametric attribute')
    else:
        is_use_pred_addattr = False
        print(Fore.GREEN + 'use label parametric attribute')

    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)

    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    # os.environ[‘CUDA_VISIBLE_DEVICES‘] 使用指定的GPU及GPU显存
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 定义数据集，训练集及对应加载器
    train_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=True)
    test_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=False)
    n_segpart = len(train_dataset.seg_names) # num_classes 数据集中的模型类别数
    print('num of segment part: ', n_segpart)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 获取分类模型
    classifier = CrossAttention_Seg(n_segpart, args.n_metatype)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print(Fore.GREEN + 'training from exist model: ' + model_savepth)
    except:
        print(Fore.GREEN + 'no existing model, training from scratch')

    if is_use_pred_addattr:
        try:
            predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
            predictor.load_state_dict(torch.load('model_trained/TriFeaPred_ValidOrig_fuse.pth'))
            predictor = predictor.eval()
            print(Fore.GREEN + 'load param attr predictor from', 'model_trained/TriFeaPred_ValidOrig_fuse.pth')
        except:
            print(Fore.GREEN + 'load param attr predictor failed')
            exit(1)

    classifier.apply(inplace_relu)
    classifier = classifier.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate, # 0.001
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate # 1e-4
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    save_fig_num = 3  # 训练集测试集中每个epoch分别保存的可视化图形数
    best_oa = -1.0

    '''TRANING'''
    start_epoch = 0
    for epoch in range(start_epoch, args.epoch):

        logstr_epoch = 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch)
        classifier = classifier.train()

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)

        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        save_fig_count = 0

        # 计算评价指标用变量
        total_correct = 0
        total_points = 0
        iou_per_part_sum = torch.zeros(n_segpart)
        iou_per_part_count = torch.zeros(n_segpart)

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):
            points = data[0].float().cuda()
            target = data[1].long().cuda()

            # 使用预测属性
            if is_use_pred_addattr:
                euler_angle, nearby, meta_type = predictor(points)
                nearby, meta_type = torch.exp(nearby), torch.exp(meta_type)
                euler_angle, nearby, meta_type = euler_angle.detach(), nearby.detach(), meta_type.detach()

            else:
                euler_angle = data[2].float().cuda()
                nearby = data[3].long().cuda()
                meta_type = data[4].long().cuda()

                # 将标签转化为 one-hot
                # <- is_nearby: [bs, npnt], meta_type: [bs, npnt]
                nearby = F.one_hot(nearby, 2)
                meta_type = F.one_hot(meta_type, args.n_metatype)

            seg_pred = classifier(points, euler_angle, nearby, meta_type)

            # 显示分割效果图
            if is_show_img:
                save_fig_step = len(trainDataLoader) // save_fig_num
                if batch_id % save_fig_step == 0:
                    save_path = os.path.join(confusion_dir, f'train-{epoch}-{save_fig_count}.png')
                    segfig_save(points, seg_pred, save_path)

                    save_path_gt = os.path.join(confusion_dir, f'train-{epoch}-{save_fig_count}-GT.png')
                    segfig_save(points, F.one_hot(target, n_segpart), save_path_gt)

                    save_fig_count += 1

            # 评价指标变量
            # Overall Accuracy (OA)
            pred_classes = seg_pred.argmax(dim=2)  # Size: [batch_size, n_points]
            total_correct += (pred_classes == target).sum().item()
            total_points += target.numel()

            # instance mean Intersection over Union (instance mIOU)
            # 对每个分割部件计算 IoU
            for part in range(n_segpart):
                intersection = ((pred_classes == part) & (target == part)).sum().item()
                union = ((pred_classes == part) | (target == part)).sum().item()
                if union > 0:  # 只在存在此部件时才计算 IoU
                    iou_per_part_sum[part] += float(intersection) / float(union)
                    iou_per_part_count[part] += 1

            # 利用loss更新参数
            target = target.view(-1, 1)[:, 0]
            seg_pred = seg_pred.contiguous().view(-1, n_segpart)
            loss = F.nll_loss(seg_pred, target)
            loss.backward()
            optimizer.step()

        # 计算该epoch的评价指标
        oa = total_correct / total_points
        for c_part in range(n_segpart):
            if abs(iou_per_part_count[c_part].item()) < 1e-6:
                # iou_per_part_count[c_part] = 1
                # iou_per_part_sum[c_part] = 0
                print('发现某个分割类别为零:', c_part)

        iou_per_part_avg = iou_per_part_sum / iou_per_part_count
        iou_per_part_avg = torch.nan_to_num(iou_per_part_avg, nan=0.0)

        miou = iou_per_part_avg.mean().item()
        logstr_trainaccu = f'train_oa\t{oa}\ttrain_miou\t{miou}'

        for c_part in range(n_segpart):
            logstr_trainaccu += f'\t{train_dataset.seg_names[c_part]}_miou\t{iou_per_part_avg[c_part]}'

        print(logstr_trainaccu.replace('\t', ' '))


        global_epoch += 1
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            classifier = classifier.eval()

            save_fig_count = 0
            total_miou = 0.0  # 累加每个 batch 的 MIOU
            total_batches = 0  # 记录 batch 的数量

            # 计算评价指标用变量
            total_correct = 0
            total_points = 0
            iou_per_part_sum = torch.zeros(n_segpart)
            iou_per_part_count = torch.zeros(n_segpart)

            for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader),smoothing=0.9):
                points = data[0].float().cuda()
                target = data[1].long().cuda()

                # 使用预测属性
                if is_use_pred_addattr:
                    euler_angle, nearby, meta_type = predictor(points)
                    nearby, meta_type = torch.exp(nearby), torch.exp(meta_type)
                    euler_angle, nearby, meta_type = euler_angle.detach(), nearby.detach(), meta_type.detach()

                else:
                    euler_angle = data[2].float().cuda()
                    nearby = data[3].long().cuda()
                    meta_type = data[4].long().cuda()

                    # 将标签转化为 one-hot
                    # <- is_nearby: [bs, npnt], meta_type: [bs, npnt]
                    nearby = F.one_hot(nearby, 2)
                    meta_type = F.one_hot(meta_type, args.n_metatype)

                seg_pred = classifier(points, euler_angle, nearby, meta_type)
                batch_miou = compute_instance_miou(seg_pred, target, n_segpart)
                total_miou += batch_miou
                total_batches += 1

                if is_show_img:
                    save_fig_step = len(testDataLoader) // save_fig_num
                    if batch_id % save_fig_step == 0:
                        save_path = os.path.join(confusion_dir, f'test-{epoch}-{save_fig_count}.png')
                        segfig_save(points, seg_pred, save_path)

                        save_path_gt = os.path.join(confusion_dir, f'test-{epoch}-{save_fig_count}-GT.png')
                        segfig_save(points, F.one_hot(target, n_segpart), save_path_gt)

                        save_fig_count += 1

                # 评价指标变量
                # Overall Accuracy (OA)
                pred_classes = seg_pred.argmax(dim=2)  # Size: [batch_size, n_points]
                total_correct += (pred_classes == target).sum().item()
                total_points += target.numel()

                # instance mean Intersection over Union (instance mIOU)
                # 对每个分割部件计算 IoU
                for part in range(n_segpart):
                    intersection = ((pred_classes == part) & (target == part)).sum().item()
                    union = ((pred_classes == part) | (target == part)).sum().item()
                    if union > 0:  # 只在存在此部件时才计算 IoU
                        iou_per_part_sum[part] += float(intersection) / float(union)
                        iou_per_part_count[part] += 1

            # 计算该epoch的评价指标
            oa = total_correct / total_points

            for c_part in range(n_segpart):
                if abs(iou_per_part_count[c_part].item()) < 1e-6:
                    # iou_per_part_count[c_part] = 1
                    # iou_per_part_sum[c_part] = 0
                    print('发现某个分割类别为零:', c_part)

            iou_per_part_avg = iou_per_part_sum / iou_per_part_count
            iou_per_part_avg = torch.nan_to_num(iou_per_part_avg, nan=0.0)
            miou = iou_per_part_avg.mean().item()

            # 计算所有 batch 的平均 MIOU
            epoch_miou = total_miou / total_batches if total_batches > 0 else 0.0

            accustr = f'test_oa\t{oa}\tseg_class_miou\t{miou}\tinstance miou\t{epoch_miou}'
            for c_part in range(n_segpart):
                accustr += f'\t{train_dataset.seg_names[c_part]}_miou\t{iou_per_part_avg[c_part]}'

            # logstr_trainaccu = ''
            logger.info(logstr_epoch + logstr_trainaccu + accustr)
            print(accustr.replace('\t', '  '))

            # 额外保存最好的模型
            if best_oa < oa:
                best_oa = oa
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    clear_log('./log')
    init(autoreset=True)
    main(parse_args())








