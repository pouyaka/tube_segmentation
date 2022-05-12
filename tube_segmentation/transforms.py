import albumentations
import albumentations.pytorch


class TargetTransform(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, mask):
        return [mask]


def augment_transform(long_edge, augmentation=True, pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.4803, 0.4770, 0.5089)
        std = (0.0009, 0.0009, 0.0007)

    augment_list = [
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=3, val_shift_limit=2, p=0.5),
    ]

    train_trans = albumentations.Compose(
        [
            albumentations.LongestMaxSize(long_edge),
            albumentations.Sequential(augment_list, p=1) if augmentation else None,
            albumentations.Normalize(mean=mean, std=std),
            albumentations.GaussNoise(var_limit=(0.4, 0.6), p=0.5) if augmentation else None,
            albumentations.pytorch.ToTensorV2(transpose_mask=True)
        ]
    )
    val_trans = albumentations.Compose(
        [
            albumentations.LongestMaxSize(long_edge),
            albumentations.Normalize(mean=mean, std=std),
            albumentations.pytorch.ToTensorV2(transpose_mask=True)
        ]
    )
    return train_trans, val_trans


def factory(args):
    train_transform, val_transform = augment_transform(args.long_edge,
                                                       augmentation=args.augmentation,
                                                       pretrained=args.pretrained)

    target_transform = TargetTransform(args.model_name)

    return train_transform, val_transform, target_transform
