import torch
import numpy as np

from train_epoch import TrainEpoch, ValidEpoch
from get_losses import get_losses
from get_model import get_model
from get_dataset import *
from image_utils import visualize_results
from other_utils import param_to_string
import torch.optim as optim
from torch.optim import lr_scheduler


def create_train_epoch_runner(model, num_classes, losses, optimizer, device, verbose, EvaluatorIn):

    train_epoch = TrainEpoch(
        model,
        num_classes=num_classes,
        losses=losses,
        optimizer=optimizer,
        device=device,
        verbose=verbose,
        EvaluatorIn=EvaluatorIn
    )
    return train_epoch


def create_valid_epoch_runner(model, num_classes, losses, device, verbose, EvaluatorIn):

    valid_epoch = ValidEpoch(
        model,
        num_classes=num_classes,
        losses=losses,
        device=device,
        verbose=verbose,
        EvaluatorIn=EvaluatorIn
    )

    return valid_epoch


def print_metrics(logs, metrics):
    str_to_print = ''

    for m in metrics:

        temp = str(np.asscalar(logs[m]))
        str_to_print = str_to_print + m + ': ' + temp + ', '

    print(str_to_print)


def train(model, optimizer, scheduler, losses, metrics, train_dataset, valid_dataset, num_epoch, args,  EvaluatorIn=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_epoch = create_train_epoch_runner(model, args['num_classes'], losses, optimizer, device, args['verbose'], EvaluatorIn)
    train_loader = torch.utils.data.DataLoader(train_dataset, args['train_batch_size'], shuffle=args['shuffle_train'], num_workers=args['num_workers'])

    if args['valid_dataset']: # training with validation
        valid_loader = torch.utils.data.DataLoader(valid_dataset, args['val_batch_size'], shuffle=False, num_workers=args['num_workers'])
        valid_epoch = create_valid_epoch_runner(model,  args['num_classes'], losses, device, args['verbose'],  EvaluatorIn)

    valid_logs = None

    for i in range(0, num_epoch):
        if args['verbose']:
            print('\nEpoch: {}/{}'.format(i+1, num_epoch))

        train_logs = train_epoch.run(train_loader)
        if args['verbose']:
            print_metrics(train_logs, metrics)

        if args['valid_dataset']:
            valid_logs = valid_epoch.run(valid_loader)
            if args['verbose']:
                if args['verbose']:
                    print_metrics(valid_logs, metrics)

        if args['visualize']:
            visualize_results(train_dataset, valid_dataset, model, device)

        scheduler.step()

    return valid_logs, train_logs


def train_normal(args, num_epoch, model_save=None, model_load=None, imgSaver=None, EvaluatorIn=None):

    """
    Normal training stage
    :param args: various parameters needed for training
    :param model_save: file to save model to
    :param model_load: file to load model from
    :return: train logs, validation logs

    """
    model = get_model(args)
    losses = get_losses(args)
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, args['scheduler_interval'], args['scheduler_gamma'])
    valid_metric = None

    if model_load is not None:
        model.load_state_dict(torch.load(model_load), strict=False)

    train_dataset, valid_dataset = get_train_val_datasets(args)

    valid_logs, train_logs = train(model,
                                   optimizer,
                                   scheduler,
                                   losses,
                                   args['metrics'],
                                   train_dataset,
                                   valid_dataset,
                                   num_epoch,
                                   args,
                                   EvaluatorIn)

    if not args['valid_dataset']:
        valid_logs = train_logs

    torch.save(model.state_dict(), model_save)

    return valid_logs, train_logs


def train_anneal(args, model_save, model_load, imgSaver=None):

    """
    Trains model based on regularized loss with annealing
    :param args: various parameters needed for training
    :param model_save: file to save model to
    :param model_load: file to load model from
    :return: train logs

    """

    print('Entering Annealing Training Stage')
    num_epochs_per_step = [args['num_epoch_per_annealing_step']] * len(args['reg_loss_params'])

    for i, param in zip(range(len(num_epochs_per_step)), args['reg_loss_params']):
        args['loss_names'][args['reg_loss_name']] = param

        if args['verbose']:
            print("\nAnnealing epoch {}/{}: params: {}".format(i+1, len(num_epochs_per_step), param_to_string(param)))

        valid_logs, train_logs = train_normal(args, num_epochs_per_step[i], model_save, model_load, imgSaver)
        model_load = model_save

    return train_logs


