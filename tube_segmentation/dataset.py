import os
import torch.utils.data
import numpy as np
from scipy.io import loadmat
import cv2


class HistologyDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mat_root, transform=None, target_transform=None, num_class=2):
        self.img_root = img_root
        self.mat_root = mat_root
        self.transform = transform
        self.target_transform = target_transform
        self.num_class = num_class
        self.ids = os.listdir(self.img_root)
        # remove .TIF from names
        self.ids = [os.path.splitext(x)[0] for x in self.ids]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.ids[index] + '.TIF')
        mat_path = os.path.join(self.mat_root, self.ids[index] + '.mat')
        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load mask
        mask = load_mask(height=image.shape[0], width=image.shape[1], mat_dir=mat_path, num_class=self.num_class)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.ids)


class ImageList(torch.utils.data.Dataset):
    def __init__(self, img_root, images=None, mat_root=None, transform=None, num_class=2):
        self.img_root = img_root
        self.mat_root = mat_root
        self.transform = transform
        self.num_class = num_class
        self.images = images or os.listdir(img_root)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.images[index])
        img_name = os.path.splitext(self.images[index])[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed_image = None
        if self.transform is not None:
            transformed = self.transform(image=image)
            transformed_image = transformed["image"]
        if self.mat_root is not None:
            mat_path = os.path.join(self.mat_root, img_name + '.mat')
            # load mask
            mask = load_mask(height=image.shape[0], width=image.shape[1], mat_dir=mat_path, num_class=self.num_class)
            transformed = self.transform(image=image, mask=mask)
            transformed_mask = transformed["mask"]
            return transformed_image, image, img_name, transformed_mask

        return transformed_image, image, img_name

    def __len__(self):
        return len(self.images)


def load_mask(height, width, mat_dir=None, num_class=2):
    matfile = loadmat(mat_dir)
    segment = matfile['segment']
    mask = np.zeros((height, width, num_class)).astype(np.float32)

    # extract label
    tube = segment[0][0]['tube_l']
    tube[tube > 1] = 1
    lumen = segment[0][0]['lumen_l']
    lumen[lumen > 1] = 1
    label = tube - lumen
    label[label == 255] = 1
    # extract background
    background = np.ones((height, width))
    # extract boundary
    boundary = _get_boundary(segment, height, width)
    # dilate boundary
    kernel = np.ones((5, 5), np.uint8)
    boundary = cv2.dilate(boundary, kernel, iterations=1)

    if num_class == 2:
        subtract = label - boundary
        subtract[subtract < 0] = 0
        mask[:, :, 1] = subtract
        background[subtract == 1] = 0
        mask[:, :, 0] = background
    else:  # 3 class: make boundary a separated class
        # background
        label_back = cv2.dilate(label, kernel, iterations=1)
        background[label_back == 1] = 0
        mask[:, :, 0] = background
        # subtract boundary from label
        subtract = label - boundary
        subtract[subtract < 0] = 0
        mask[:, :, 1] = subtract
        mask[:, :, 2] = boundary
    return mask


def _get_boundary(seg_mat, h, w):
    boundary = np.zeros((h, w))
    for im in range(seg_mat[0][0]['tube_b'].shape[0]):
        # tube boundary
        tube_b = seg_mat[0][0]['tube_b'] - 1
        # for x
        tube_b_x = tube_b[im][0][0]
        # for y
        tube_b_y = tube_b[im][0][1]

        for i, j in zip(tube_b_x, tube_b_y):
            boundary[i, j] = 1

    for im in range(seg_mat[0][0]['lumen_b'].shape[0]):
        # lumen boundary
        lumen_b = seg_mat[0][0]['lumen_b'] - 1
        # for x
        lumen_b_x = lumen_b[im][0][0]
        # for y
        lumen_b_y = lumen_b[im][0][1]

        for i, j in zip(lumen_b_x, lumen_b_y):
            boundary[i, j] = 1
    return boundary


def train_factory(args, train_transform=None, val_transform=None, target_transform=None):
    train_dataset = HistologyDataset(img_root=args.image_dir, mat_root=args.mat_dir,
                                     transform=train_transform, target_transform=target_transform,
                                     num_class=args.num_class)
    val_dataset = HistologyDataset(img_root=args.image_dir, mat_root=args.mat_dir,
                                   transform=val_transform, target_transform=target_transform,
                                   num_class=args.num_class)
    return train_dataset, val_dataset
