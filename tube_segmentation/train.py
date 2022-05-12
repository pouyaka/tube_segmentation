"""train a network"""
import argparse
import datetime
import logging
import sys
import torch

from . import dataset, transforms, losses, network
from .trainer import Trainer


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', default='data/imgs/',
                        help='images directory')
    parser.add_argument('--mat-dir', default='data/mats/',
                        help='mat files directory')
    parser.add_argument('--num-class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--long-edge', default=1280, type=int,
                        help='longest edge of image(to resize with maintaining aspect ratio)')
    parser.add_argument('--no-augmentation', dest='augmentation',
                        default=True, action='store_false',
                        help='do not apply data augmentation')
    parser.add_argument('--model-name', default='unet',
                        help='network, e.g. proposed method: unetscse')
    parser.add_argument('--pretrained', dest='pretrained',
                        default=False, action='store_true',
                        help='apply pretrained weights(Imagenet)')
    parser.add_argument('--tta-apply', dest='tta_apply',
                        default=False, action='store_true',
                        help='apply test time augmentation')
    parser.add_argument('--checkpoint', default=None,
                        help='Load a pretrained model from a checkpoint.')
    parser.add_argument('--loss-fn', default=['ce'], nargs='+',
                        help='loss functions: ce,mse,dice,dicece,iou,focal')
    parser.add_argument('--epochs', default=60, type=int,
                        help='number of epochs to train')
    parser.add_argument('--train-batch-size', default=4, type=int,
                        help='train batch size')
    parser.add_argument('--val-batch-size', default=2, type=int,
                        help='batch size')
    parser.add_argument('--k-folds', default=5, type=int,
                        help='number of folds for k-fold cross validation')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    args = parser.parse_args()
    args.output = output_name(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def output_name(args):
    out = 'outputs/{}-{}-{}'.format(args.model_name, '-'.join(args.loss_fn), str(args.num_class)+'class')
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out += '-{}'.format(now)
    return out


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
    net_cpu, start_epoch = network.factory(args)
    model = net_cpu.to(args.device)
    optim = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
    loss_list = losses.factory(args)
    train_transform, val_transform, target_transform = transforms.factory(args)
    train_dataset, val_dataset = dataset.train_factory(args, train_transform, val_transform, target_transform)

    # getting same results
    torch.manual_seed(7)
    # Define the trainer
    trainer = Trainer(
        model, loss_list, optim,
        lr_scheduler=scheduler,
        stride_apply=args.stride_apply,
        tta_apply=args.tta_apply,
        device=args.device,
        model_name=args.model_name,
        output_name=args.output,
        seed_lr=args.lr
    )
    trainer.train_loop(train_dataset, val_dataset, epochs=args.epochs, start_epoch=start_epoch,
                       train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
                       folds_num=args.k_folds)


if __name__ == '__main__':
    main()
