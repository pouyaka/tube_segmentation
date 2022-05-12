import torch.utils.data
import torch.nn.functional
import torch
import sklearn.model_selection
import time
import logging
import numpy as np

from .metrics import ConfusionMatrix, dice, jaccard, fscore, accuracy


class Trainer(object):
    def __init__(self, model, losses, optimizer,
                 lr_scheduler=None, stride_apply=1, tta_apply=False,
                 device=None, model_name=None, output_name=None, seed_lr=1e-4):
        self.model = model
        self.model_name = model_name
        self.losses = losses
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.stride_apply = stride_apply
        self.device = device
        self.tta_apply = tta_apply
        self.output_name = output_name
        self.seed_lr = seed_lr
        self.log = logging.getLogger(self.__class__.__name__)

    def train_loop(self, train_data, val_data, epochs, start_epoch=0, train_batch_size=4,
                   val_batch_size=1, folds_num=5):
        # Define the K-fold Cross Validator
        kfolder = sklearn.model_selection.KFold(n_splits=folds_num, shuffle=True, random_state=7)
        data_tp = 0
        data_fp = 0
        data_tn = 0
        data_fn = 0
        pretrained_weights = self.model.state_dict()

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfolder.split(train_data)):
            self.log.info(f'FOLD {fold + 1}')
            self.log.info(np.array(val_data.ids)[val_ids])

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            # Define data loaders for training and validation data in this fold
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=train_batch_size, pin_memory=True, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=val_batch_size, pin_memory=True, sampler=val_subsampler)

            # reset weights for new fold
            self.model.load_state_dict(pretrained_weights)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.seed_lr

            for epoch in range(start_epoch, epochs):
                self.train(train_loader, epoch)
                epoch_tp, epoch_fp, epoch_tn, epoch_fn = self.val(val_loader, epoch)
                if self.lr_scheduler:
                    self.lr_scheduler.step()
            data_tp += epoch_tp
            data_fp += epoch_fp
            data_tn += epoch_tn
            data_fn += epoch_fn
            self.write_model(fold + 1, epochs)
        # metrics
        dice_sc = dice(confusion_matrix=[data_tp, data_fp, data_tn, data_fn])
        jaccard_sc = jaccard(confusion_matrix=[data_tp, data_fp, data_tn, data_fn])
        f_sc = fscore(confusion_matrix=[data_tp, data_fp, data_tn, data_fn])
        acc = accuracy(confusion_matrix=[data_tp, data_fp, data_tn, data_fn])
        self.log.info(f'dice_score: {dice_sc},jaccard_score: {jaccard_sc},f_score: {f_sc},accuracy: {acc}')

    def train(self, scenes, epoch):
        start_time = time.time()
        self.model.train()
        epoch_loss = 0.0
        last_batch_end = time.time()
        self.optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(scenes):
            preprocess_time = time.time() - last_batch_end

            batch_start = time.time()
            apply_gradients = ((batch_idx + 1) % self.stride_apply == 0) or ((batch_idx + 1) == len(scenes))
            loss = self.train_batch(data, target, apply_gradients)

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss

            batch_time = time.time() - batch_start

            last_batch_end = time.time()

        epoch_info = {
            'train_epoch': epoch + 1,
            'n_batches': len(scenes),
            'lr': self.lr(),
            'train_loss': round(float(epoch_loss.item()) / len(scenes), 2),
            'train_time': round(time.time() - start_time, 1),
        }
        self.log.info(epoch_info)

    def val(self, scenes, epoch):
        start_time = time.time()

        # Train mode implies outputs are for losses, so have to use it here.
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        epoch_loss = 0.0
        epoch_tp = 0.0
        epoch_fp = 0.0
        epoch_tn = 0.0
        epoch_fn = 0.0
        for data, target in scenes:
            loss, tp, fp, tn, fn = self.val_batch(data, target)
            # update confusion matrix for each epoch
            epoch_tp += tp
            epoch_fp += fp
            epoch_tn += tn
            epoch_fn += fn

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss

        jaccard_sc = jaccard(confusion_matrix=[epoch_tp, epoch_fp, epoch_tn, epoch_fn])
        f_sc = fscore(confusion_matrix=[epoch_tp, epoch_fp, epoch_tn, epoch_fn])
        d_sc = dice(confusion_matrix=[epoch_tp, epoch_fp, epoch_tn, epoch_fn])

        eval_time = time.time() - start_time

        epoch_info = {
            'val_epoch': epoch + 1,
            'n_batches': len(scenes),
            'val_loss': round(float(epoch_loss.item()) / len(scenes), 2),
            'jaccard_score': round(jaccard_sc, 2),
            'F_score': round(f_sc, 2),
            'dice': round(d_sc, 2),
            'val_time': round(eval_time, 1),
        }
        self.log.info(epoch_info)
        return epoch_tp, epoch_fp, epoch_tn, epoch_fn

    def train_batch(self, data, targets, apply_gradients=True):
        data, targets = self.to_device(data, targets)

        # train network
        outputs = self.model(data)
        outputs = [outputs]
        loss, losses_list = self.compute_loss(outputs, targets)
        if loss is not None:
            loss.backward()
        if apply_gradients:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def val_batch(self, data, targets):
        data, targets = self.to_device(data, targets)

        with torch.no_grad():
            outputs = self.model(data)
            outputs = [outputs]
            loss, _ = self.compute_loss(outputs, targets)
            if self.tta_apply:
                confusion_matrix_batch = ConfusionMatrix(self.tta(data), targets[0])
            else:
                confusion_matrix_batch = ConfusionMatrix(torch.nn.functional.softmax(outputs[0], dim=1), targets[0])
            tp, fp, tn, fn = confusion_matrix_batch.get_matrix()

        return loss, tp, fp, tn, fn

    def compute_loss(self, outputs, targets):
        loss = None
        losses_list = [l(o, t) for l, o, t in zip(self.losses, outputs, targets)]
        if losses_list:
            loss = sum(losses_list)
        return loss, losses_list
    
    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def to_device(self, data, targets):
        if self.device:
            data = data.to(self.device, non_blocking=True)
            targets = [t.to(self.device, non_blocking=True) for t in targets]
        return data, targets

    def write_model(self, fold, epoch):
        self.model.cpu()

        model = self.model

        filename = self.output_name + f'-fold-{fold}.pt'
        torch.save({
            'model': model,
            'epoch': epoch
        }, filename)
        self.model.to(self.device)
        
    def tta(self, image):
        aug_batch = torch.cat(
            [
                image,
                image.flip(2),
                image.flip(3)
            ],
            dim=0
        )
        outs = self.model(aug_batch)
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
