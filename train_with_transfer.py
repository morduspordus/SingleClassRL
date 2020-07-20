import os
from get_standard_arguments import get_standard_arguments
from get_losses import *
from other_utils import create_file_name
from training_utils import train_anneal, train_normal


def two_stage_training(dataset_path, model_load, model_name, dataset_name, im_size):
    training_type = '_transfer_'

    args = get_standard_arguments(model_name, dataset_name, im_size, dataset_path)

    output_dir = ('./run')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    add_to_file_path, model_save = create_file_name(dataset_name, model_name, im_size, training_type, output_dir)

    if args['negative_class']:  # using negative loss
        losses = standard_complete_loss_with_negative(reg_weight=args['reg_weight'],
                                                      vol_min=args['vol_min'],
                                                      vol_weight_s=args['vol_weight_s'],
                                                      neg_weight=args['negative_weight'])
    else:
        losses = standard_complete_loss(reg_weight=args['reg_weight'],
                                        vol_min=args['vol_min'],
                                        vol_weight_s=args['vol_weight_s'])

    args['reg_loss_name'], args['loss_names'] = losses.loss_names()

    num_epoch = 50

    print('Taining with transfer from ' + model_load)

    train_normal(args, num_epoch, model_save, model_load)


if __name__ == "__main__":

    #model_name = "UMobV2"
    model_name = "UResNext"

    # dataset_name = 'OxfordPet'
    # dataset_path = 'D:/Olga/data/Oxford_iit_pet'  # path to oxford pet dataset

    #model_load = './trained_models/OxfordPet_UMobV2_128_from_anneal_V1.pt'
    model_load = './trained_models/OxfordPet_UResNext_128_from_anneal_V1.pt'

    dataset_name = 'DUT'
    dataset_path = 'D:/Olga/data/Saliency/DUT-OMRON'

    # dataset_name = 'ECSSD'
    # dataset_path = 'D:/Olga/data/Saliency/ECSSD'
    #

    two_stage_training(dataset_path, model_load, model_name, dataset_name, im_size=128)


