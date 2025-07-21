'''
train constraint prediction
'''
import os
import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse

from data_utils.ParamDataLoader import STEPMillionDataLoader
from models.cstpnt import CstPnt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2500, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_primitive', type=int, default=4, help='number of considered meta type')
    parser.add_argument('--workers', type=int, default=10, help='dataloader workers')
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_pack1', help='root of dataset')

    args = parser.parse_args()
    return args


def main(args):
    save_str = 'TriFeaPred_ValidOrig'

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_dataset = STEPMillionDataLoader(root=args.root_dataset, npoints=args.num_point, data_augmentation=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))  # , drop_last=True

    '''MODEL LOADING'''
    predictor = CstPnt(n_points_all=args.num_point, n_primitive=args.n_primitive).cuda()

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        predictor.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    if not args.use_cpu:
        predictor = predictor.cuda()

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.learning_rate, # 0.001
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate # 1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    num_batch = len(train_dataloader)

    '''TRANING'''
    for epoch in range(args.epoch):
        print(f'current epoch: {epoch}/{args.epoch}')
        predictor = predictor.train()

        for batch_id, data in enumerate(train_dataloader, 0):
            xyz, eula_angle_label, nearby_label, meta_type_label = data
            bs, n_points, _ = xyz.size()
            n_items_batch = bs * n_points

            if not args.use_cpu:
                xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()
            else:
                xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float(), eula_angle_label.float(), nearby_label.long(), meta_type_label.long()

            optimizer.zero_grad()
            pred_eula_angle, pred_edge_nearby, pred_meta_type = predictor(xyz)

            # vis_pointcloudattr(point_set[0, :, :].detach().cpu().numpy(), np.argmax(pred_meta_type[0, :, :].detach().cpu().numpy(), axis=1))

            loss_eula = F.mse_loss(eula_angle_label, pred_eula_angle)

            pred_edge_nearby = pred_edge_nearby.contiguous().view(-1, 2)
            nearby_label = nearby_label.view(-1)
            loss_nearby = F.nll_loss(pred_edge_nearby, nearby_label)

            pred_meta_type = pred_meta_type.contiguous().view(-1, args.n_primitive)
            meta_type_label = meta_type_label.view(-1)
            loss_metatype = F.nll_loss(pred_meta_type, meta_type_label)

            loss_all = loss_eula + loss_nearby + loss_metatype

            loss_all.backward()
            optimizer.step()

            # accu
            choice_nearby = pred_edge_nearby.data.max(1)[1]
            correct_nearby = choice_nearby.eq(nearby_label.data).cpu().sum()
            choice_meta_type = pred_meta_type.data.max(1)[1]
            correct_meta_type = choice_meta_type.eq(meta_type_label.data).cpu().sum()

            log_str = f'train_loss\t{loss_all.item()}\teula_loss\t{loss_eula.item()}\tnearby_loss\t{loss_nearby.item()}\tmetatype_loss\t{loss_metatype.item()}\tnearby_accu\t{correct_nearby.item() / float(n_items_batch)}\tmeta_type_accu\t{correct_meta_type.item() / float(n_items_batch)}'
            logger.info(log_str)

            print_str = f'[{epoch}: {batch_id}/{num_batch}] train loss: {loss_all.item()}, eula loss: {loss_eula.item()}, nearby loss: {loss_nearby.item()},metatype loss: {loss_metatype.item()}, nearby accu: {correct_nearby.item() / float(n_items_batch)}, meta type accu: {correct_meta_type.item() / float(n_items_batch)}'
            print(print_str)

        scheduler.step()
        torch.save(predictor.state_dict(), model_savepth)


if __name__ == '__main__':
    main(parse_args())

