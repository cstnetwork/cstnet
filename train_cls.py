
"""
train classification
"""

import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm

from data_utils.ParamDataLoader import MCBDataLoader
from models.cstnet_s2 import CstNetS2
from models.cstpnt import CstPnt
from models.utils import all_metric_cls


def parse_args():

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--n_primitive', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--n_point', type=int, default=2000, help='Point Number')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A', help='root of dataset')

    return parser.parse_args()


def main(args):
    # parameters
    save_str = 'ca_final_predattr'
    is_use_pred_addattr = True

    # logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # datasets
    train_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.n_point, is_train=True, data_augmentation=False)
    test_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.n_point, is_train=False, data_augmentation=False)
    num_class = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # loading model
    classifier = CstNetS2(num_class, args.n_primitive)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    if is_use_pred_addattr:
        try:
            predictor = CstPnt(n_points_all=args.n_point, n_primitive=args.n_primitive).cuda()
            predictor.load_state_dict(torch.load('model_trained/cstpnt_abc25t.pth'))
            predictor = predictor.eval()
            print('load param attr predictor from', 'model_trained/cstpnt_abc25t.pth')
        except:
            print('load param attr predictor failed')
            exit(1)

    classifier = classifier.cuda()

    # optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # training
    for epoch in range(args.epoch):
        logstr_epoch = 'Epoch %d/%d:' % (epoch + 1, args.epoch)
        all_preds = []
        all_labels = []

        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            points = data[0].cuda()
            target = data[1].long().cuda()

            if is_use_pred_addattr:
                mad, adj, pt = predictor(points)
                adj, pt = torch.exp(adj), torch.exp(pt)
                mad, adj, pt = mad.detach(), adj.detach(), pt.detach()

            else:
                mad = data[2].float().cuda()
                adj = data[3].long().cuda()
                pt = data[4].long().cuda()

                # <- adj: [bs, npnt], pt: [bs, npnt]
                adj = F.one_hot(adj, 2)
                pt = F.one_hot(pt, args.n_primitive)

            optimizer.zero_grad()

            pred = classifier(points, mad, adj, pt)
            loss = F.nll_loss(pred, target)

            loss.backward()
            optimizer.step()

            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        all_metric_train = all_metric_cls(all_preds, all_labels)

        logstr_trainaccu = f'\ttrain_instance_accu:\t{all_metric_train[0]}\ttrain_class_accu:\t{all_metric_train[1]}\ttrain_F1_Score:\t{all_metric_train[2]}\tmAP\t{all_metric_train[3]}'
        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            all_preds = []
            all_labels = []

            classifier = classifier.eval()
            for j, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = data[0].cuda()
                target = data[1].long().cuda()

                if is_use_pred_addattr:
                    mad, adj, pt = predictor(points)
                    adj, pt = torch.exp(adj), torch.exp(pt)
                    mad, adj, pt = mad.detach(), adj.detach(), pt.detach()

                else:
                    mad = data[2].float().cuda()
                    adj = data[3].long().cuda()
                    pt = data[4].long().cuda()

                    adj = F.one_hot(adj, 2)
                    pt = F.one_hot(pt, args.n_primitive)

                pred = classifier(points, mad, adj, pt)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

            all_metric_test = all_metric_cls(all_preds, all_labels)
            accustr = f'\ttest_instance_accuracy\t{all_metric_test[0]}\ttest_class_accuracy\t{all_metric_test[1]}\ttest_F1_Score\t{all_metric_test[2]}\tmAP\t{all_metric_test[3]}'

            print(accustr)
            logger.info(logstr_epoch + logstr_trainaccu + accustr)


if __name__ == '__main__':
    main(parse_args())

