
"""
train classification
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
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from data_utils.ParamDataLoader import MCBDataLoader
from data_utils.ParamDataLoader import save_confusion_mat
from models.cstnet_cls import CstNet_Cls
from models.cst_pred import TriFeaPred_OrigValid


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A', help='root of dataset')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def accuracy_over_class(all_labels, all_preds, n_classes):
    accuracies = []

    for class_idx in range(n_classes):
        class_mask = (all_labels == class_idx)
        if class_mask.sum().item() == 0:
            continue
        class_accuracy = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
        accuracies.append(class_accuracy)

    return np.mean(accuracies)


def main(args):
    save_str = 'ca_final_predattr'
    is_use_pred_addattr = True

    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)

    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 定义数据集，训练集及对应加载器
    train_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=True, data_augmentation=False)
    test_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=False, data_augmentation=False)
    num_class = len(train_dataset.classes)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    classifier = CstNet_Cls(num_class, args.n_metatype)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    if is_use_pred_addattr:
        try:
            predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
            predictor.load_state_dict(torch.load('model_trained/TriFeaPred_ValidOrig_fuse.pth'))
            predictor = predictor.eval()
            print('load param attr predictor from', 'model_trained/TriFeaPred_ValidOrig_fuse.pth')
        except:
            print('load param attr predictor failed')
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_accu = -1.0

    '''TRANING'''
    start_epoch = 0
    for epoch in range(start_epoch, args.epoch):
        logstr_epoch = 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch)
        mean_correct = []
        classifier = classifier.train()

        pred_cls = []
        target_cls = []

        total_samples = torch.zeros(num_class).cuda()
        positive_samples = torch.zeros(num_class).cuda()

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):
            points = data[0].cuda()
            target = data[1].long().cuda()

            if is_use_pred_addattr:
                eula_angle_label, nearby_label, meta_type_label = predictor(points)
                nearby_label, meta_type_label = torch.exp(nearby_label), torch.exp(meta_type_label)
                eula_angle_label, nearby_label, meta_type_label = eula_angle_label.detach(), nearby_label.detach(), meta_type_label.detach()

            else:
                eula_angle_label = data[2].float().cuda()
                nearby_label = data[3].long().cuda()
                meta_type_label = data[4].long().cuda()

                # <- is_nearby: [bs, npnt], meta_type: [bs, npnt]
                nearby_label = F.one_hot(nearby_label, 2)
                meta_type_label = F.one_hot(meta_type_label, args.n_metatype)

            optimizer.zero_grad()

            pred = classifier(points, eula_angle_label, nearby_label, meta_type_label)
            loss = F.nll_loss(pred, target)

            loss.backward()
            optimizer.step()
            global_step += 1

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            for class_idx in range(num_class):
                total_samples[class_idx] += (target == class_idx).sum()

                predicted_classes = torch.argmax(pred, dim=1)
                positive_samples[class_idx] += ((target == class_idx) & (predicted_classes == class_idx)).sum()

            pred_cls += pred_choice.tolist()
            target_cls += target.tolist()

        save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'train-{epoch}.png'))

        all_preds = torch.Tensor(pred_cls)
        all_labels = torch.Tensor(target_cls)

        acc_over_class = accuracy_over_class(all_labels, all_preds, num_class)

        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()
        macro_f1_score = f1_score(all_labels, all_preds, average='weighted')

        train_instance_acc = np.mean(mean_correct)
        logstr_trainaccu = f'\ttrain_instance_accu:\t{train_instance_acc}\ttrain_class_accu:\t{acc_over_class}\ttrain_F1_Score:\t{macro_f1_score}'
        scheduler.step()
        global_epoch += 1
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            total_correct = 0
            total_testset = 0

            classifier = classifier.eval()

            pred_cls = []
            target_cls = []

            all_preds = []
            all_labels = []

            total_samples = torch.zeros(num_class).cuda()
            positive_samples = torch.zeros(num_class).cuda()

            for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points = data[0].cuda()
                target = data[1].long().cuda()

                if is_use_pred_addattr:
                    eula_angle_label, nearby_label, meta_type_label = predictor(points)
                    nearby_label, meta_type_label = torch.exp(nearby_label), torch.exp(meta_type_label)
                    eula_angle_label, nearby_label, meta_type_label = eula_angle_label.detach(), nearby_label.detach(), meta_type_label.detach()

                else:
                    eula_angle_label = data[2].float().cuda()
                    nearby_label = data[3].long().cuda()
                    meta_type_label = data[4].long().cuda()

                    nearby_label = F.one_hot(nearby_label, 2)
                    meta_type_label = F.one_hot(meta_type_label, args.n_metatype)

                pred = classifier(points, eula_angle_label, nearby_label, meta_type_label)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]

                for class_idx in range(num_class):
                    total_samples[class_idx] += (target == class_idx).sum()

                    predicted_classes = torch.argmax(pred, dim=1)
                    positive_samples[class_idx] += ((target == class_idx) & (predicted_classes == class_idx)).sum()

                pred_cls += pred_choice.tolist()
                target_cls += target.tolist()

            save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'eval-{epoch}.png'))

            all_preds = np.vstack(all_preds)
            all_labels = np.hstack(all_labels)

            all_labels_bin = label_binarize(all_labels, classes=np.arange(num_class))

            ap_scores = []
            for i in range(num_class):
                ap = average_precision_score(all_labels_bin[:, i], all_preds[:, i])
                ap_scores.append(ap)

            mAP = np.mean(ap_scores)

            acc_over_instance = total_correct / float(total_testset)

            all_preds = torch.Tensor(pred_cls)
            all_labels = torch.Tensor(target_cls)

            acc_over_class = accuracy_over_class(all_labels, all_preds, num_class)

            all_preds = all_preds.numpy()
            all_labels = all_labels.numpy()
            macro_f1_score = f1_score(all_labels, all_preds, average='weighted')

            accustr = f'\ttest_instance_accuracy\t{acc_over_instance}\ttest_class_accuracy\t{acc_over_class}\ttest_F1_Score\t{macro_f1_score}\tmAP\t{mAP}'
            print(accustr)
            logger.info(logstr_epoch + logstr_trainaccu + accustr)

            if best_instance_accu < acc_over_class:
                best_instance_accu = acc_over_class
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    args = parse_args()
    main(args)

