from oxford_iii_pet import OxfordPet


def get_dataset(args, split='train'):

    dataset_to_return = OxfordPet(args, split)

    return dataset_to_return


def get_train_val_datasets(args):

    train_dataset = get_dataset(args, split=args['split'])

    if args['valid_dataset']:
        val_dataset = get_dataset(args, split='val')
    else:
        val_dataset = None

    return train_dataset, val_dataset


def get_val_dataset(args, split='val'):

    val_dataset = get_dataset(args, split=split)

    return val_dataset
