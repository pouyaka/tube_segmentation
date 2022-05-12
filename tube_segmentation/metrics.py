import torch
import torch.nn.functional
import numpy as np


def assert_shape(output, target):

    assert output.shape == target.shape, "Shape mismatch: {} and {}".format(
        output.shape, target.shape)


def threshold_mask(probs):
    labels = torch.argmax(probs, dim=1)
    predicted_mask = torch.nn.functional.one_hot(labels)
    return predicted_mask


class ConfusionMatrix:

    def __init__(self, output=None, target=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.target_empty = None
        self.target_full = None
        self.output_empty = None
        self.output_full = None
        self.target = target
        self.mask = threshold_mask(output)
        self.output = output
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.output_empty = None
        self.output_full = None
        self.target_empty = None
        self.target_full = None

    def compute(self):

        if self.mask is None or self.target is None:
            raise ValueError("'output' and 'target' must both be set to compute confusion matrix.")

        assert_shape(self.mask, self.target)

        self.tp = ((self.mask != 0) & (self.target != 0)).sum().item()
        self.fp = ((self.mask != 0) & (self.target == 0)).sum().item()
        self.tn = ((self.mask == 0) & (self.target == 0)).sum().item()
        self.fn = ((self.mask == 0) & (self.target != 0)).sum().item()
        self.size = int(np.prod(self.target.size(), dtype=np.int64))
        self.output_empty = not torch.any(self.mask)
        self.output_full = torch.all(self.mask)
        self.target_empty = not torch.any(self.target)
        self.target_full = torch.all(self.target)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.output_empty, self.output_full, self.target_empty, self.target_full):
            if case is None:
                self.compute()
                break

        return self.output_empty, self.output_full, self.target_empty, self.target_full


def dice(output=None, target=None, confusion_matrix=None, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float((2*tp) / ((2*tp) + fp + fn))


def jaccard(output=None, target=None, confusion_matrix=None, **kwargs):  # iou
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float(tp / (tp + fp + fn))


def precision(output=None, target=None, confusion_matrix=None, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float(tp / (tp + fp))


def sensitivity(output=None, target=None, confusion_matrix=None, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float(tp / (tp + fn))


def recall(output=None, target=None, confusion_matrix=None, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(output, target, confusion_matrix, **kwargs)


def specificity(output=None, target=None, confusion_matrix=None, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float(tn / (tn + fp))


def accuracy(output=None, target=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(output, target)
        tp, fp, tn, fn = confusion_matrix.get_matrix()
    else:
        tp = confusion_matrix[0]
        fp = confusion_matrix[1]
        tn = confusion_matrix[2]
        fn = confusion_matrix[3]

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(output=None, target=None, confusion_matrix=None, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(output, target, confusion_matrix)
    recall_ = recall(output, target, confusion_matrix)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)
