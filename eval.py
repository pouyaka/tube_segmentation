""" evaluate metrics on the dataset"""

import argparse
import torch
import sys
import os
import logging
import torch.utils.data
import time
import torch.nn.functional
import albumentations
import albumentations.pytorch

from . import network, transforms, dataset
from .metrics import ConfusionMatrix, dice, jaccard, fscore


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', default='new data/imgs/',
                        help='images directory')
    parser.add_argument('--mat-dir', default='new data/mats/',
                        help='mat files directory')
    parser.add_argument('--num-class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--long-edge', default=1280, type=int,
                        help='longest edge of image(to resize with maintaining aspect ratio)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default=False, action='store_true',
                        help='apply pretrained weights(Imagenet)')
    parser.add_argument('--tta-apply', dest='tta_apply',
                        default=False, action='store_true',
                        help='apply test time augmentation')
    parser.add_argument('--checkpoint', default=None,
                        help='Load a pretrained model from a checkpoint.')
    parser.add_argument('--batch-size', default=3, type=int,
                        help='batch size')
    args = parser.parse_args()
    args.output = os.path.splitext(args.checkpoint)[0]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def configure_logs(args):
    format_ = logging.Formatter('%(asctime)s-%(name)s:%(message)s', datefmt='%y/%m/%d-%H:%M:%S')

    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(format_)

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(format_)

    logging.basicConfig(level=logging.INFO, handlers=[c_handler, file_handler])

    logging.info({
        'argv': sys.argv,
        'args': vars(args)
    })


def main():
    args = cli()
    configure_logs(args)
    net_cpu, _ = network.factory(args)
    model = net_cpu.to(args.device)
    val_transform = default_transform(args.long_edge, imagenet=args.pretrained)
    val_data = dataset.HistologyDataset(img_root=args.image_dir, mat_root=args.mat_dir,
                                        transform=val_transform, num_class=args.num_class)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, pin_memory=True)
    start_time = time.time()

    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0
    for data, targets in val_loader:
        data = data.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        with torch.no_grad():
            if args.tta_apply:
                confusion_matrix_batch = ConfusionMatrix(tta(data, model), targets)
            else:
                outputs = model(data)
                confusion_matrix_batch = ConfusionMatrix(torch.nn.functional.softmax(outputs, dim=1), targets)
            tp, fp, tn, fn = confusion_matrix_batch.get_matrix()
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    jaccard_sc = jaccard(confusion_matrix=[total_tp, total_fp, total_tn, total_fn])
    f_sc = fscore(confusion_matrix=[total_tp, total_fp, total_tn, total_fn])
    d_sc = dice(confusion_matrix=[total_tp, total_fp, total_tn, total_fn])

    eval_time = time.time() - start_time

    eval_info = {
        'n_batches': len(val_loader),
        'jaccard_score': round(jaccard_sc, 4),
        'F_score': round(f_sc, 4),
        'dice': round(d_sc, 4),
        'val_time': round(eval_time, 1),
    }
    logging.info(eval_info)


def tta(data, model):
    aug_batch = torch.cat(
        [
            data,
            data.flip(2),
            data.flip(3)
        ],
        dim=0
    )
    outs = model(aug_batch)
    outs = torch.softmax(outs, dim=1)
    orig, flip_ud, flip_lr = torch.chunk(outs, 3)
    deaug_batch = torch.stack(
        [
            orig,
            flip_ud.flip(2),
            flip_lr.flip(3)
        ]
    )
    average = deaug_batch.mean(dim=0)
    return average


def default_transform(long_edge, imagenet=True):
    if imagenet:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.4803, 0.4770, 0.5089)
        std = (0.0009, 0.0009, 0.0007)

    trans = albumentations.Compose(
        [
            albumentations.LongestMaxSize(long_edge),
            albumentations.Normalize(mean=mean, std=std),
            albumentations.pytorch.ToTensorV2()
        ]
    )
    return trans


if __name__ == '__main__':
    main()
