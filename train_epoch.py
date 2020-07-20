# Modified from segmentation-models-pytorch V 0.0.3

import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from evaluator import Evaluator


class Epoch:

    def __init__(self, model, num_classes, losses, stage_name,  device='cpu', verbose=True, EvaluatorIn=None):
        self.model = model
        self.losses = losses
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self._to_device()

        if EvaluatorIn is None:
            self.evaluator = Evaluator(num_classes, device)
        else:
            self.evaluator = EvaluatorIn

    def _to_device(self):
        for loss in self.losses:
            loss.to(self.device)
        self.model.to(self.device)

    @classmethod
    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        losses_meters = {loss.__name__: AverageValueMeter() for loss in self.losses}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for sample in iterator:
                x = sample['image']
                y = sample['label']
                if 'image_class' in sample:
                    image_class = sample['image_class']
                else:
                    image_class = None
                x, y = x.to(self.device), y.to(self.device)
                loss, losses, y_pred = self.batch_update(x, y, image_class, sample)

                # update losses logs

                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'total_loss': loss_meter.mean}
                logs.update(loss_logs)

                for loss_fn in self.losses:
                    loss_value = loss_fn(y_pred, y, x, image_class)
                    if type(loss_value) == torch.Tensor:
                        loss_value = loss_value.cpu().detach().numpy()
                    losses_meters[loss_fn.__name__].add(loss_value)
                losses_logs = {k: v.mean for k, v in losses_meters.items()}
                logs.update(losses_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)


        logs['acc'] = self.evaluator.Pixel_Accuracy()
        logs['fscore'] = self.evaluator.f_score()
        logs['jaccard']  = self.evaluator.Jaccard()
        logs['miou']   = self.evaluator.Mean_Intersection_over_Union()
        logs['acc_class'] = self.evaluator.Pixel_Accuracy_Class()

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, num_classes, losses, optimizer, device='cpu', verbose=True, EvaluatorIn=None):
        super(TrainEpoch, self).__init__(
            model=model,
            num_classes=num_classes,
            losses=losses,
            stage_name='train',
            device=device,
            verbose=verbose,
            EvaluatorIn=EvaluatorIn
        )
        self.optimizer = optimizer

    def on_epoch_start(self):

        self.model.train()
        self.evaluator.reset()

    def batch_update(self, x, y, image_class, sample):

        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        losses = {loss.__name__: loss(prediction, y, x, image_class) for loss in self.losses}

        loss = 0.0

        for (k, v) in losses.items():
            loss = loss + v

        loss.backward()

        self.optimizer.step()
        self.evaluator.add_batch(y, prediction)

        return loss, losses, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, num_classes, losses, device='cpu', verbose=True, EvaluatorIn=None):
        super(ValidEpoch, self).__init__(
            model=model,
            num_classes=num_classes,
            losses=losses,
            stage_name='valid',
            device=device,
            verbose=verbose,
            EvaluatorIn=EvaluatorIn
        )

    def on_epoch_start(self):
        self.model.eval()
        self.evaluator.reset()

    def batch_update(self, x, y, image_class, sample):
        with torch.no_grad():
            prediction = self.model.forward(x)

            losses = {loss.__name__: loss(prediction, y, x, image_class) for loss in self.losses}

            loss = 0.0
            for (k, v) in losses.items():
                loss = loss + v

            self.evaluator.add_batch(y, prediction)

        return loss, losses, prediction


