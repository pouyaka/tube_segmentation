import torch
import torch.nn
import torch.nn.functional


def loss_weight(n_class):
    if n_class == 2:
        weights = [1.25423722, 0.83146071]
    else:
        weights = [0.89716412, 0.5875235, 5.45502627]
    weights = torch.FloatTensor(weights)
    return weights


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        probs = torch.nn.functional.softmax(inputs, dim=1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        return 1 - dice


class DiceCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        dice_loss = DiceLoss()(inputs, targets)
        ce = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='mean')
        dice_ce = ce + dice_loss

        return dice_ce


class TverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.3, beta=0.7):
        inputs = torch.nn.functional.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return 1 - tversky


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.3, beta=0.7, gamma=2):
        inputs = torch.nn.functional.softmax(inputs, dim=1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        focal_tversky = (1 - tversky) ** gamma

        return focal_tversky


def factory(args):
    losses = []
    for loss in args.loss_fn:
        if loss == 'ce':
            losses.append(torch.nn.CrossEntropyLoss())
        elif loss == 'wce':
            class_weight = loss_weight(args.num_class)
            losses.append(torch.nn.CrossEntropyLoss(weight=class_weight.to(args.device)))
        elif loss == 'mse':
            losses.append(torch.nn.MSELoss())
        elif loss == 'dice':
            losses.append(DiceLoss())
        elif loss == 'dicece':
            losses.append(DiceCELoss())
        elif loss == 'dicewce':
            class_weight = loss_weight(args.num_class)
            losses.append(DiceCELoss(weight=class_weight.to(args.device)))
        elif loss == 'tversky':
            losses.append(TverskyLoss())
        elif loss == 'focaltversky':
            losses.append(FocalTverskyLoss())
        else:
            raise Exception('unknown loss type {}'.format(loss))
    return [l.to(device=args.device) for l in losses]
