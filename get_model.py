from unet_models import *


def get_model(args):

    if args['model_name'] == "UMobV2":
        model = Unet_MobileNetV2(args['num_classes'],
                                 args['use_fixed_features'],
                                 final_activation=args['final_activation'],
                                 num_final_features=args['num_final_features']
                                 )

    elif args['model_name'] == "UResNext":
        model = Unet_se_resnext50_32x4d(args['num_classes'],
                                 args['use_fixed_features'],
                                 final_activation=args['final_activation'],
                                 num_final_features=args['num_final_features']
                                 )

    else:
        print('Model {} not available.'.format(args.model_name))
        raise NotImplementedError

    return model


