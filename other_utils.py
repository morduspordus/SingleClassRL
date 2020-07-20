import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os


def model_file_name(dataset_name, model_name, valid_metric, model_load, model_params, dataset_params):

    smooth_kernel = model_params["smooth_kernel"]
    image_height = dataset_params["image_height"]
    valid_metric = "{0:.2f}".format(valid_metric * 100)

    if model_load == None:
        from_model = "None"
    elif model_load == "Anneal":
        from_model = "Anneal"
    else:
        x = model_load.split("\\")
        x = x[2]
        y = x.split('.pt')
        from_model = y[0]

    if len(from_model) > 40:
        from_model = from_model[:40]

    if smooth_kernel == None:
        kernel_str = "_nosmooth_"
    else:
        kernel_str ="_smooth" + str(smooth_kernel)+"_"

    model_weights_name = os.path.join(WEIGHTS_DIR,  dataset_name  +str(image_height)+"_" + model_name + kernel_str + '_M_' + valid_metric +  '_FFOM_' + from_model + '.pt')

    return model_weights_name


def create_file_name(dataset_name, model_name, im_size, training_type, output_dir):

    version = 1
    add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
    model_save = os.path.join(output_dir, add_to_file_path + '.pt')

    while os.path.exists(model_save):
        version += 1
        add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
        model_save = os.path.join(output_dir, add_to_file_path + '.pt')

    return add_to_file_path, model_save


def param_to_string(param):
    str_params = ['{} - {:.4}'.format(k, v) for k, v in param.items()]
    s = ', '.join(str_params)
    return s

