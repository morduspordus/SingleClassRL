from param_functions import sparse_crf_anneal_param, sparse_crf_to_list_dict
from get_losses import *


def get_standard_arguments(model_name, dataset_name, im_size, dataset_path):
    """
        Sets up most of the parameters used throughout the project
    """
    args = dict()

    args['d_path'] = dataset_path
    args['verbose'] = True
    args['visualize'] = True      # visualizes sample images after each training epoch
    args['valid_dataset'] = True  # after each training epoch, run validation epoch

    args['model_name'] = model_name
    args['dataset_name'] = dataset_name

    # parameters for loss function
    args['negative_class'] = False
    args['reg_loss_name'], args['loss_names'] = standard_complete_loss().loss_names()
    args['vol_min'] = 0.15         # object is encouraged to be of normalized area at least vol_min
    args['vol_weight_s'] = 5.      # weight of the second term in volume loss corresponding to vol_min
    args['reg_weight'] = 100.      # weight of regularization loss
    args['negative_weight'] = 2.   # weight of negative loss, if using

    # parameters for annealing training stage
    args['reg_loss_params'] = sparse_crf_to_list_dict(sparse_crf_anneal_param)
    args['num_epoch_per_annealing_step'] = 1

    # parameters for dataset and images
    args['split'] = 'train'              # use the 'train' split for training
    args['ignore_class'] = 255
    args['cats_only'] = False            # setting to True will ignore dog images
    args['dogs_only'] = False            # setting to True will ignore cat images
    args['cats_dogs_separate'] = False   # setting to True will give cats and dogs as separate classes, do not use
    args['dogs_negative'] = False        # set to  True if wish to train on cats and use dogs as negative class
    args['cats_negative'] = False        # set to  True if wish to train on dogs and use cats as negative class
    args['num_classes'] = 2

    args['im_size'] = im_size
    args['base_size'] = args['im_size']
    args['crop_size'] = args['im_size']
    args['image_normalize_mean'] = (0.485, 0.456, 0.406)
    args['image_normalize_std'] = (0.229, 0.224, 0.225)

    # Parameters for CNN
    args['use_fixed_features'] = True
    args['final_activation'] = 'softmax'
    args['num_final_features'] = 64

    # parameters for optimizer and sheduler
    args['scheduler_interval'] = 100
    args['scheduler_gamma'] = 0.1
    args['learning_rate'] = 0.001

    # parameters for training
    args['shuffle_train'] = True
    args['num_workers'] = 4
    args['train_batch_size'] = 16
    args['val_batch_size'] = 16

    args['metrics'] = {'fscore', 'acc', 'miou', 'jaccard'}

    return args
