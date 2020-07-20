import torch
import torch.nn as nn
import loss_utils as UF


class SparseCRFLoss(nn.Module):
    __name__ = 'sparseCRF'

    def __init__(self, weight, sigma, subtract_eps, mean, std, num_classes, negative_class=False, diag=False):
        super(SparseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma = sigma
        self.subtract_eps = subtract_eps
        self.diag = diag
        self.negative_class = negative_class
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

    def forward(self, y_pr, y_gt, img, true_class):
        mask_h, mask_v = UF.compute_edge_mask(img, self.sigma)

        mask_h = mask_h - self.subtract_eps
        mask_v = mask_v - self.subtract_eps

        loss = 0.
        # regularized loss is not applied to the background class
        for ch in range(1, self.num_classes):
            loss = loss + UF.regularized_loss_per_channel(mask_h, mask_v, ch, y_pr, true_class, self.negative_class)

        if self.diag:
            mask_d1, mask_d2 = UF.compute_edge_mask_diag(img, self.sigma)
            mask_d1 = mask_d1 - self.subtract_eps
            mask_d2 = mask_d2 - self.subtract_eps
            loss = loss + UF.regularized_loss_per_channel_diag(mask_d1, mask_d2, 1, y_pr, true_class,
                                                               self.negative_class)

        return self.weight * loss


class MiddleSqLoss(nn.Module):
    __name__ = 'middle_sqL'

    def __init__(self, weight, square_w, num_classes):
        super(MiddleSqLoss, self).__init__()
        self.weight = weight
        self.square_w = square_w
        self.num_classes = num_classes

    def forward(self, y_pr, y_gt, img, true_class):
        loss = 0

        for ch in range(1, self.num_classes):  # do not compute this loss for the background
            loss = loss + UF.middle_sq_loss_per_channel(ch, y_pr, true_class, self.square_w)

        return self.weight * loss


class VolumeLoss(nn.Module):
    __name__ = 'volumeL'

    # cl is the channel for which take volume loss
    def __init__(self, weight, fraction, cl, negative_class=False, weight_s=1., vol_min=0.15):
        super(VolumeLoss, self).__init__()
        self.weight = weight
        self.weight_s = weight_s
        self.fraction = fraction
        self.cl = cl
        self.negative_class = negative_class
        self.vol_min = vol_min

    def forward(self, y_pr, y_gt, img, true_class):

        if self.negative_class:
            # extract samples whose true class is not 0, i.e. all but negative samples
            y_pr = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_not_equal_fn)
            if y_pr is None:
                return 0.

        samples_volume = torch.mean(y_pr[:, self.cl, :, :], dim=(1, 2))

        small_volumes = torch.min(samples_volume, torch.tensor(self.vol_min).cuda())
        loss_s = torch.mean((small_volumes - self.vol_min) ** 2) * self.weight_s

        loss = self.weight * ((torch.mean(samples_volume) - self.fraction) ** 2)

        return loss + loss_s


class BorderLoss(nn.Module):
    # is only applied to the background class, i.e channel 0
    __name__ = 'borderL'

    def __init__(self, weight, border_w, cl):
        super(BorderLoss, self).__init__()
        self.weight = weight
        self.border_w = border_w
        self.cl = cl

    def forward(self, y_pr, y_gt, img, true_class):
        left = torch.mean(y_pr[:, self.cl, 0:self.border_w, :])
        right = torch.mean(y_pr[:, self.cl, -self.border_w:, :])
        top = torch.mean(y_pr[:, self.cl, :, 0:self.border_w])
        bottom = torch.mean(y_pr[:, self.cl, :, -self.border_w:])

        loss = ((left - 1.0) ** 2 + (right - 1.0) ** 2 + (top - 1.0) ** 2 + (bottom - 1.0) ** 2) / 4.0

        return self.weight * loss


class NegativeClassLoss(nn.Module):
    __name__ = 'neg_clL'

    def __init__(self, weight):
        super(NegativeClassLoss, self).__init__()
        self.weight = weight

    def forward(self, y_pr, y_gt, img, true_class):

        # extract negative class samples
        negative_samples = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_equal_fn)

        if negative_samples is not None:
            loss = torch.mean(negative_samples[:, 1, :, :])  # the channel corresponding to object should be 0
            loss = loss ** 2
        else:
            loss = 0.

        return self.weight * loss

