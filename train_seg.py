
"""
train segmentation
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init

from data_utils.ParamDataLoader import Seg360GalleryDataLoader
from data_utils.ParamDataLoader import segfig_save
from models.cstnet_seg import CrossAttention_Seg
from models.cst_pred import TriFeaPred_OrigValid


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', default=1, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')

    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number')

    parser.add_argument('--save_str', type=str, default='ca_final_predattr_part_seg', help='---')
    parser.add_argument('--is_show_img', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--is_use_pred_addattr', type=str, default='True', choices=['True', 'False'], help='---')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\360Gallery_Seg', help='root of dataset')

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
    batch_size = seg_pred.shape[0]

    seg_pred_label = torch.argmax(seg_pred, dim=-1)  # [batch_size, n_points]

    iou_list = []

    for b in range(batch_size):
        iou_per_part = []
        for part in range(n_seg_part):
            pred_mask = (seg_pred_label[b] == part)
            target_mask = (target[b] == part)

            intersection = torch.sum(pred_mask & target_mask).item()
            union = torch.sum(pred_mask | target_mask).item()

            if union == 0:
                continue
            else:
                iou = intersection / union
                iou_per_part.append(iou)

        if iou_per_part:
            instance_miou = sum(iou_per_part) / len(iou_per_part)
            iou_list.append(instance_miou)

    if iou_list:
        return sum(iou_list) / len(iou_list)
    else:
        return 0.0


def clear_log(log_dir):
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
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

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=True)
    test_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=False)
    n_segpart = len(train_dataset.seg_names)
    print('num of segment part: ', n_segpart)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
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
    save_fig_num = 3
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

        total_correct = 0
        total_points = 0
        iou_per_part_sum = torch.zeros(n_segpart)
        iou_per_part_count = torch.zeros(n_segpart)

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):
            points = data[0].float().cuda()
            target = data[1].long().cuda()

            if is_use_pred_addattr:
                euler_angle, nearby, meta_type = predictor(points)
                nearby, meta_type = torch.exp(nearby), torch.exp(meta_type)
                euler_angle, nearby, meta_type = euler_angle.detach(), nearby.detach(), meta_type.detach()

            else:
                euler_angle = data[2].float().cuda()
                nearby = data[3].long().cuda()
                meta_type = data[4].long().cuda()

                # <- is_nearby: [bs, npnt], meta_type: [bs, npnt]
                nearby = F.one_hot(nearby, 2)
                meta_type = F.one_hot(meta_type, args.n_metatype)

            seg_pred = classifier(points, euler_angle, nearby, meta_type)

            if is_show_img:
                save_fig_step = len(trainDataLoader) // save_fig_num
                if batch_id % save_fig_step == 0:
                    save_path = os.path.join(confusion_dir, f'train-{epoch}-{save_fig_count}.png')
                    segfig_save(points, seg_pred, save_path)

                    save_path_gt = os.path.join(confusion_dir, f'train-{epoch}-{save_fig_count}-GT.png')
                    segfig_save(points, F.one_hot(target, n_segpart), save_path_gt)

                    save_fig_count += 1

            # Overall Accuracy (OA)
            pred_classes = seg_pred.argmax(dim=2)  # Size: [batch_size, n_points]
            total_correct += (pred_classes == target).sum().item()
            total_points += target.numel()

            # instance mean Intersection over Union (instance mIOU)
            for part in range(n_segpart):
                intersection = ((pred_classes == part) & (target == part)).sum().item()
                union = ((pred_classes == part) | (target == part)).sum().item()
                if union > 0:
                    iou_per_part_sum[part] += float(intersection) / float(union)
                    iou_per_part_count[part] += 1

            target = target.view(-1, 1)[:, 0]
            seg_pred = seg_pred.contiguous().view(-1, n_segpart)
            loss = F.nll_loss(seg_pred, target)
            loss.backward()
            optimizer.step()

        oa = total_correct / total_points
        for c_part in range(n_segpart):
            if abs(iou_per_part_count[c_part].item()) < 1e-6:
                # iou_per_part_count[c_part] = 1
                # iou_per_part_sum[c_part] = 0
                print('zero instance class:', c_part)

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
            total_miou = 0.0
            total_batches = 0

            total_correct = 0
            total_points = 0
            iou_per_part_sum = torch.zeros(n_segpart)
            iou_per_part_count = torch.zeros(n_segpart)

            for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader),smoothing=0.9):
                points = data[0].float().cuda()
                target = data[1].long().cuda()

                if is_use_pred_addattr:
                    euler_angle, nearby, meta_type = predictor(points)
                    nearby, meta_type = torch.exp(nearby), torch.exp(meta_type)
                    euler_angle, nearby, meta_type = euler_angle.detach(), nearby.detach(), meta_type.detach()

                else:
                    euler_angle = data[2].float().cuda()
                    nearby = data[3].long().cuda()
                    meta_type = data[4].long().cuda()

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

                # Overall Accuracy (OA)
                pred_classes = seg_pred.argmax(dim=2)  # Size: [batch_size, n_points]
                total_correct += (pred_classes == target).sum().item()
                total_points += target.numel()

                # instance mean Intersection over Union (instance mIOU)
                for part in range(n_segpart):
                    intersection = ((pred_classes == part) & (target == part)).sum().item()
                    union = ((pred_classes == part) | (target == part)).sum().item()
                    if union > 0:
                        iou_per_part_sum[part] += float(intersection) / float(union)
                        iou_per_part_count[part] += 1

            oa = total_correct / total_points

            for c_part in range(n_segpart):
                if abs(iou_per_part_count[c_part].item()) < 1e-6:
                    # iou_per_part_count[c_part] = 1
                    # iou_per_part_sum[c_part] = 0
                    print('zero instance class:', c_part)

            iou_per_part_avg = iou_per_part_sum / iou_per_part_count
            iou_per_part_avg = torch.nan_to_num(iou_per_part_avg, nan=0.0)
            miou = iou_per_part_avg.mean().item()

            epoch_miou = total_miou / total_batches if total_batches > 0 else 0.0

            accustr = f'test_oa\t{oa}\tseg_class_miou\t{miou}\tinstance miou\t{epoch_miou}'
            for c_part in range(n_segpart):
                accustr += f'\t{train_dataset.seg_names[c_part]}_miou\t{iou_per_part_avg[c_part]}'

            # logstr_trainaccu = ''
            logger.info(logstr_epoch + logstr_trainaccu + accustr)
            print(accustr.replace('\t', '  '))

            if best_oa < oa:
                best_oa = oa
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    clear_log('./log')
    init(autoreset=True)
    main(parse_args())


