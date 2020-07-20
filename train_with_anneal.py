import os
from get_standard_arguments import get_standard_arguments
from get_losses import *
from other_utils import create_file_name
from training_utils import train_anneal, train_normal


def two_stage_training(model_name, dataset_name, dataset_path, im_size):

    training_type = 'anneal'
    model_load = None

    args = get_standard_arguments(model_name, dataset_name, im_size, dataset_path)

    output_dir = ('./run')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    add_to_file_path, model_save = create_file_name(dataset_name, model_name, im_size, training_type, output_dir)

    losses = standard_complete_loss(vol_min=args['vol_min'], vol_weight_s=args['vol_weight_s'])

    args['reg_loss_name'], args['loss_names'] = losses.loss_names()

    train_anneal(args, model_save, model_load)

    model_load = model_save
    training_type_new = 'from_anneal'
    model_save = str.replace(model_save, training_type, training_type_new)

    num_epoch = 50
    losses.reset_reg_weight(args['reg_weight'])
    args['reg_loss_name'], args['loss_names'] = losses.loss_names()

    print('Entering Normal Training Stage')

    train_normal(args, num_epoch, model_save, model_load)


if __name__ == "__main__":
    dataset_name = 'OxfordPet'
    model_name = "UMobV2" # another model option is "UResNext"

    dataset_path = './data/Oxford_iit_pet'  # path to oxford pet dataset

    two_stage_training(model_name, dataset_name, dataset_path, im_size=128)




