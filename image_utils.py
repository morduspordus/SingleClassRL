import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from PIL import Image
import torch

from get_dataset import get_dataset, get_val_dataset
from get_model import get_model


"""
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def transform_to_image(inp, mean_image, std_image):

    """ Undoes changes in the mean and standard deviation """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean_image)
    std = np.array(std_image)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp


def visualize_images(**images):
    """PLot images in one row."""
    plt.close()
    n = len(images)
    plt.figure(figsize=(12, 3))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show(block=False)
    plt.pause(1 / 20)


def process_visualize_image(dataset, device, model):

    n = np.random.choice(len(dataset))

    sample = dataset[n]
    image = sample['image']
    gt_mask = sample['label']

    x_tensor = image.to(device).unsqueeze(0)

    pr = model(x_tensor)

    if isinstance(pr, list):
        pr = pr[0]

    _, pr_mask = torch.max(pr, dim=1)
    pr_mask = pr_mask.squeeze()

    image_original = img_denormalize(image.cpu().numpy(), dataset.mean, dataset.std)
    object_channel = pr[0, 1, :, :]
    object_channel = object_channel.detach()

    return image_original, pr_mask, gt_mask, object_channel


def visualize_results(train_dataset, valid_dataset, model, device):
    model.eval()

    if valid_dataset is None:
        valid_dataset = train_dataset

    image_original, pr_mask, gt_mask, pr = process_visualize_image(train_dataset, device, model)

    gt_out = np.array(gt_mask).astype(np.uint8)

    dataset_name = train_dataset.name()
    segmap_gt = decode_segmap(gt_out)


    if valid_dataset is not None:
        image_original_val, pr_mask_val, gt_mask_val, pr = process_visualize_image(valid_dataset, device, model)
        gt_val_out = np.array(gt_mask_val).astype(np.uint8)
        segmap_gt_val = decode_segmap(gt_val_out)

        visualize_images(
            image_train = image_original,
            gt_train=segmap_gt,
            predict_train=decode_segmap(pr_mask.cpu().numpy()),
            image_valid=image_original_val,
            gt_val=segmap_gt_val,
            predict_valid = decode_segmap(pr_mask_val.cpu().numpy()),
           # pr=pr.cpu().numpy()
        )


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 3
    label_colours = get_oxford_pet_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_oxford_pet_labels():
    """Load the mapping that associates oxfordpet classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def img_denormalize(img, mean, std):
    """Denormalize image
    Returns:
        denormalized  image
    """

    img = np.transpose(img, axes=[1, 2, 0])
    img *= std
    img += mean
    img *= 255.0
    img = img.astype(np.uint8)

    return img

