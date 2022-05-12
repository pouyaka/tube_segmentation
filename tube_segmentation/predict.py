""" predict for given image(s)"""

import cv2
import os
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import matplotlib.pyplot as plt
import argparse
import albumentations
import albumentations.pytorch

from . import dataset, network


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', default='data/imgs/',
                        help='image(s) directory')
    parser.add_argument('--images', nargs='+', default=[],
                        help='image(s) name(if you want all of images in folder let it be None)')
    parser.add_argument('--mat-dir', default=None,
                        help='mats directory')
    parser.add_argument('--num-class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--long-edge', default=1600, type=int,
                        help='longest edge of image(to resize with maintaining aspect ratio)')
    parser.add_argument('--no-show', dest='show',
                        default=True, action='store_false',
                        help='do not show the result')
    parser.add_argument('--tta', dest='tta',
                        default=False, action='store_true',
                        help='apply test time augmentation')
    parser.add_argument('--checkpoint', default=None,
                        help='Load a pretrained model from a checkpoint.')
    parser.add_argument('--imagenet-norm', dest='imagenet',
                        default=False, action='store_true',
                        help='normalize with imagenet mean and variance')
    parser.add_argument('-o', '--output-dir', default='outputs/',
                        help='Output directory')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.pin_memory = True
    return args


def main():
    args = cli()
    # load model
    model, _ = network.factory(args)
    model.to(args.device)

    transform = default_transform(args.long_edge, imagenet=args.imagenet)
    resize_transform = albumentations.LongestMaxSize(args.long_edge)
    data = dataset.ImageList(args.image_dir, images=args.images, mat_root=args.mat_dir,
                             transform=transform, num_class=args.num_class)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, pin_memory=args.pin_memory)
    if args.mat_dir is None:
        for batch, (preprocessed_imgs, org_imgs, names) in enumerate(data_loader):
            fin_mask_predicts = predict(preprocessed_imgs, model, args.device,
                                        tta_apply=args.tta)

            # unbatch
            for image, name, pred_mask in zip(org_imgs, names, fin_mask_predicts):
                transformed = resize_transform(image=image.numpy())
                image = transformed["image"]
                list_dict = [{'pic': image, 'label': 'image'}]
                overlay = image_overlay(image, pred_mask, n_class=args.num_class)
                list_dict.append({'pic': overlay, 'label': 'overlaid predict'})
                list_dict.append({'pic': make_gt(pred_mask), 'label': 'predicted mask'})
                make_fig(list_dict, output_dir=os.path.join(args.output_dir, name), show=args.show)
    else:
        for batch, (preprocessed_imgs, org_imgs, names, masks) in enumerate(data_loader):
            fin_mask_predicts = predict(preprocessed_imgs, model, args.device,
                                        tta_apply=args.tta)

            # unbatch
            for image, name, pred_mask, mask in zip(org_imgs, names, fin_mask_predicts, masks):
                transformed = resize_transform(image=image.numpy())
                image = transformed["image"]
                list_dict = [{'pic': image, 'label': 'image'}]
                overlay = image_overlay(image, pred_mask, n_class=args.num_class)
                list_dict.append({'pic': overlay, 'label': 'overlaid predict'})
                list_dict.append({'pic': make_gt(pred_mask), 'label': 'predicted mask'})
                list_dict.append({'pic': make_gt(mask), 'label': 'ground truth'})
                make_fig(list_dict, output_dir=os.path.join(args.output_dir, name), show=args.show)


def predict(preprocessed_imgs, model, device, tta_apply=True):
    imgs = preprocessed_imgs.to(device, non_blocking=True)
    with torch.no_grad():
        if tta_apply:
            probabilities = tta(imgs, model)
        else:
            outputs = model(imgs)
            probabilities = torch.softmax(outputs, dim=1)

        labels = torch.argmax(probabilities, dim=1)
        predicted_masks = torch.nn.functional.one_hot(labels)
        predicted_masks = predicted_masks.cpu().numpy()  # [n, h, w, n_class]
        predicted_masks = np.transpose(predicted_masks, (0, 3, 1, 2))  # [n, n_class, h, w]
    return predicted_masks


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
            albumentations.pytorch.ToTensorV2(transpose_mask=True)
        ]
    )
    return trans


def image_overlay(image, mask, n_class):
    label = np.zeros_like(image)
    if n_class == 2:
        mask = mask[1, :, :] * 255
        # extract boundary from mask by canny
        mask_2d = mask.astype('uint8')  # [h, w]
        # canny and dilation
        edges = cv2.Canny(mask_2d, threshold1=30, threshold2=100)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # make boundary red
        label[edges == 255, :] = [255, 0, 0]
    else:
        mask = np.transpose(mask, (1, 2, 0)) * 255
        # boundary is available in output of model
        # bound_mask = mask[:, :, 2]  # [h, w]
        mask_2d = mask[:, :, 1].astype('uint8')  # [h, w]
        bound_mask = cv2.Canny(mask_2d, threshold1=30, threshold2=100)
        kernel = np.ones((5, 5), np.uint8)
        bound_mask = cv2.dilate(bound_mask, kernel, iterations=1)
        # make boundary red
        label[bound_mask == 255, :] = [255, 0, 0]

    alpha = 0.6
    beta = 1.0 - alpha
    final_img = np.uint8(alpha * label + beta * image)
    return final_img


def make_gt(mask, colors=None):
    if colors is None:
        colors = [[255, 213, 90],  # background
                  [109, 212, 126],  # tubules
                  [69, 80, 115]]  # boundaries
    gr_truth = np.zeros((3, mask.shape[1], mask.shape[2]))
    for i in range(mask.shape[0]):
        for j in range(3):
            gr_truth[j, mask[i, :, :] == 1] = colors[i][j] / 255
    return np.transpose(gr_truth, (1, 2, 0))


def make_fig(ls_dict, output_dir=None, show=True):
    fig, axs = plt.subplots(1, len(ls_dict), frameon=False)

    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0)

    for i, dict_ in enumerate(ls_dict):
        subimage(axs[i], dict_['pic'], dict_['label'])

    plt.savefig(output_dir + 'predict.jpg', dpi=400, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()


def subimage(ax, img, title):
    ax.imshow(
        img,
        vmin=img.min(), vmax=img.max()
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    ax.title.set_y(-0.3)
    ax.title.set_fontfamily('Times New Roman')
    ax.title.set_fontsize(10)


def tta(image, model):
    aug_batch = torch.cat(
        [
            image,
            image.flip(2),
            image.flip(3)
        ],
        dim=0
    )
    outs = model(aug_batch)  # [0]
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


if __name__ == '__main__':
    main()
