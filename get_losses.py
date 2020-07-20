from losses import *

def get_losses(args):
    losses = []
    names_dict = args['loss_names']

    if 'negative_class_loss' in names_dict:
        negative_class = True
    else:
        negative_class = False

    for name, param in names_dict.items():

        if name == 'sparseCRF':
            if 'weight' in param:
                weight = param['weight']
            else:
                weight = 1.0
            if 'sigma' in param:
                sigma = param['sigma']
            else:
                sigma = 0.1
            if 'subtract_eps' in param:
                subtract_eps = param['subtract_eps']
            else:
                subtract_eps = 0.0
            if 'diag' in param:
                diag = param['diag']
            else:
                diag = False
            losses.append(SparseCRFLoss(weight=weight,
                                        sigma=sigma,
                                        subtract_eps=subtract_eps,
                                        mean=args['image_normalize_mean'],
                                        std=args['image_normalize_std'],
                                        num_classes=args['num_classes'],
                                        negative_class=negative_class,
                                        diag=diag))

        elif name == 'volume_loss':
            if 'weight' in param:
                weight = param['weight']
            else:
                weight = 1.0
            if 'weight_s' in param:
                weight_s = param['weight_s']
            else:
                weight_s = 1.0
            if 'fraction' in param:
                fraction = param['fraction']
            else:
                fraction = 0.5
            if 'cl' in param:
                cl = param['cl']
            else:
                cl = 1
            if 'vol_min' in param:
                vol_min = param['vol_min']
            else:
                vol_min = 0.15

            losses.append(VolumeLoss(weight=weight, fraction=fraction, cl=cl, negative_class=negative_class,
                                     weight_s=weight_s,  vol_min=vol_min))

        elif name == 'middle_sq_loss':
            if 'weight' in param:
                weight = param['weight']
            else:
                weight = 1.0
            if 'square_w' in param:
                square_w = param['square_w']
            else:
                square_w = 2
            losses.append(MiddleSqLoss(weight=weight, square_w=square_w, num_classes=args['num_classes']))

        elif name == 'border_loss':
            if 'weight' in param:
                weight = param['weight']
            else:
                weight = 1.0
            if 'border_w' in param:
                border_w = param['border_w']
            else:
                border_w = 2
            if 'cl' in param:
                cl = param['cl']
            else:
                cl = 1
            losses.append(BorderLoss(weight=weight, border_w=border_w, cl=cl))

        elif name == 'negative_class_loss':
            if 'weight' in param:
                weight = param['weight']
            else:
                weight = 1.
            losses.append(NegativeClassLoss(weight=weight))

    return losses


class standard_complete_loss():
    def __init__(self,
                 reg_weight=100,
                 vol_weight=1.0,
                 with_diag=False,
                 subtract_eps=0.0,
                 vol_weight_s=1.0,
                 sigma=0.15,
                 vol_min=0.15  # min object volume
                 ):

        super(standard_complete_loss, self).__init__()

        self.reg_weight = reg_weight
        self.vol_weight = vol_weight
        self.with_diag = with_diag
        self.subtract_eps = subtract_eps
        self.vol_weight_s = vol_weight_s
        self.sigma = sigma
        self.vol_min = vol_min

    def reset_reg_weight(self, reg_weight):
        self.reg_weight = reg_weight

    def reset_vol_weight_s(self, vol_weight_s):
        self.vol_weight_s = vol_weight_s

    def loss_names(self):

        names = {'sparseCRF': {'weight': self.reg_weight, 'sigma': self.sigma,
                                    'subtract_eps': self.subtract_eps,
                                    'diag': self.with_diag},
                  'volume_loss': {'weight': self.vol_weight,
                                  'fraction': 0.5,
                                  'cl': 1,
                                  'weight_s': self.vol_weight_s,
                                  'vol_min': self.vol_min},
                  'middle_sq_loss': {'weight': 1.0, 'square_w': 2},
                  'border_loss': {'weight': 1.0, 'border_w': 3, 'cl': 0}
                  }

        return 'sparseCRF', names



class standard_complete_loss_with_negative(standard_complete_loss):
    def __init__(self,
                 reg_weight=100,
                 vol_weight=1.0,
                 with_diag=False,
                 subtract_eps=0.0,
                 vol_weight_s=1.0,
                 sigma=0.15,
                 vol_min=0.15,
                 neg_weight=2.
                 ):
        super(standard_complete_loss_with_negative, self).__init__(
                reg_weight,
                 vol_weight,
                 with_diag,
                 subtract_eps,
                 vol_weight_s,
                 sigma,
                 vol_min)

        self.neg_weight = neg_weight

    def loss_names(self):
        _, names = super(standard_complete_loss_with_negative, self).loss_names()
        names['negative_class_loss'] = {'weight': self.neg_weight}

        return 'sparseCRF', names


