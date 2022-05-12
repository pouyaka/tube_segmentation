
import torch
import segmentation_models_pytorch as smp


def factory(args):
    if not args.checkpoint:
        epoch = 0
        pretrained = 'imagenet' if args.pretrained else None
        attention = 'scse' if 'scse' in args.model_name else None
        if 'unet' in args.model_name:
            net = smp.Unet(encoder_name='resnet34', encoder_weights=pretrained,
                           classes=args.num_class, decoder_attention_type=attention)
        elif 'deeplab' in args.model_name:
            net = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet',
                                    classes=args.num_class)
        elif 'pspnet' in args.model_name:
            net = smp.PSPNet(encoder_name='resnet34', encoder_weights='imagenet',
                             classes=args.num_class)
        else:
            raise Exception(f'unknown network {args.model_name}')
    else:
        checkpoint = torch.load(args.checkpoint)
        net = checkpoint['model']
        epoch = checkpoint['epoch']
        net.eval()
    return net, epoch
