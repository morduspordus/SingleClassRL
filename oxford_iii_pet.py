from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
import xml.dom.minidom


class OxfordPet(Dataset):
    """
    Oxford III Pet
        if args['cats_dog_separate'] = True, labels are 0: background, 1: cats, 2: dogs, 255: ignore
        if args['cats_dog_separate'] = False, labels are 0: background, 1: cats, 1: dogs, 255: ignore
        originally, in the images of this dataset, ignore_class is 3, background is 2, animal is 1
    """

    NUM_CLASSES = 3
    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param split: train/val/custom
                if split is 'custom', args['split_images_file'] provides the names of the split images
        """
        super(OxfordPet, self).__init__()

        self._base_dir = args['d_path']
        self.split = [split]
        self.args = args
        self.mean = args['image_normalize_mean']
        self.std = args['image_normalize_std']

        if split == 'train':
            if args['cats_only']:
                image_names_files =  os.path.join(self._base_dir,'annotations\\cat_train.txt')
                split_char = '\n'
            elif args['dogs_only']:
                image_names_files = os.path.join(self._base_dir, 'annotations\\dog_train.txt')
                split_char = '\n'
            else:
                image_names_files = os.path.join( self._base_dir,'annotations\\trainval.txt')

                split_char = ' '
        elif split == 'val':
            if args['cats_only']:
                image_names_files =  os.path.join(self._base_dir,'annotations\\cat_val.txt')
                split_char = '\n'
            elif args['dogs_only']:
                image_names_files = os.path.join(self._base_dir, 'annotations\\dog_val.txt')
                split_char = '\n'
            else:
                image_names_files = os.path.join(self._base_dir, 'annotations\\test.txt')
                split_char = ' '
        elif split == 'custom':
            image_names_files = os.path.join(self._base_dir, args['split_images_file'])
            split_char = ' '
        else:
            print('Split option {} is not available.'.format(split))
            raise NotImplementedError

        self.xml_files_path = os.path.join(self._base_dir, 'annotations\\xmls')

        image_files_path = os.path.join(self._base_dir, 'images')
        mask_files_path = os.path.join(self._base_dir, 'annotations\\trimaps')

        self.img_names = [line.split(split_char)[0] + '.jpg' for line in open(image_names_files)]
        self.img_paths = [image_files_path + '\\' + line.split(split_char)[0] + '.jpg' for line in open(image_names_files)]
        self.mask_paths = [mask_files_path + '\\' + line.split(split_char)[0] + '.png' for line in open(image_names_files)]

        self.image_class = [1] * len(self.img_names)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_paths)))


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        img_name = self.img_names[index]

        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                sample = self.transform_tr(sample)
                sample['name'] = self.img_names[index]
            elif split == 'val':
                sample =  self.transform_val(sample)
                sample['name'] = self.img_names[index]
            elif split == 'custom':
                sample = self.transform_tr(sample)
                sample['name'] = self.img_names[index]
            else:
                print('split {} not available.'.format(split))
                raise NotImplementedError


        _target = sample['label']
        _target[_target == 3] = 255 # change Oxford void to the standard void=255
        _target[_target == 2] = 0  # change background to 0

        if self.args['cats_dogs_separate'] or self.args['dogs_negative'] or self.args['cats_negative']:

            xml_file_name = os.path.join(self._base_dir, self.xml_files_path, img_name).replace('.jpg','.xml')
            _target = sample['label']

            if os.path.exists(xml_file_name):
                doc = xml.dom.minidom.parse(xml_file_name)
                items = doc.getElementsByTagName('name')
                animal = items[0].firstChild.data
            else:
                animal = self.name_based_animal(img_name)

            if self.args['cats_dogs_separate']:
                if animal == 'dog':
                    _target[_target==1] = 2  # dog is class 2
                    sample['image_class'] = 2
                    return sample
                else:
                    _target[_target==1] = 1  # cat is class 1
                    sample['image_class'] = 1
                    return sample
            else:
                if self.args['cats_negative'] and animal == 'cat':
                    _target_nonzero = (_target != 0)
                    _target[_target_nonzero] = 0
                    sample['image_class'] = 0
                    return sample
                elif self.args['dogs_negative'] and animal == 'dog':
                    _target_nonzero = (_target != 0)
                    _target[_target_nonzero] = 0
                    sample['image_class'] = 0
                    return sample
                else:
                    sample['image_class'] = 1
                    return sample

        else:
                sample['image_class'] = 1



        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.img_paths[index]).convert('RGB')
        _target = Image.open(self.mask_paths[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(size=self.args['crop_size']),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args['crop_size']),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Oxford_Pet(split=' + str(self.split) + ')'

    def name(self):
        return('OxfordPet')

    def name_based_animal(self, img_name):

        if 'Egyptian_Mau' in img_name:
            animal = 'cat'
        elif 'Bengal' in img_name:
            animal = 'cat'
        elif 'Ragdoll' in img_name:
            animal = 'cat'
        elif 'Abyssinian' in img_name:
            animal = 'cat'
        elif 'samoyed' in img_name:
            animal = 'dog'
        elif 'saint_bernard' in img_name:
            animal = 'dog'
        elif 'shiba_inu' in img_name:
            animal = 'dog'
        elif 'Persian' in img_name:
            animal = 'cat'
        elif 'pug' in img_name:
            animal = 'dog'
        elif 'havanese' in img_name:
            animal = 'dog'
        elif 'terrier' in img_name:
            animal = 'dog'
        elif 'chihuahua' in img_name:
            animal = 'dog'
        elif 'german_shorthaired' in img_name:
            animal = 'dog'
        elif 'Maine_Coon' in img_name:
            animal = 'cat'
        elif 'basset_hound' in img_name:
            animal = 'dog'
        elif 'american_bulldog' in img_name:
            animal = 'dog'
        elif 'great_pyrenees' in img_name:
            animal = 'dog'
        elif 'english' in img_name:
            animal = 'dog'
        elif 'Russian_Blue' in img_name:
            animal = 'cat'
        elif 'newfoundland' in img_name:
            animal = 'dog'
        elif 'boxer' in img_name:
            animal = 'dog'
        elif 'British_Shorthair' in img_name:
            animal = 'cat'
        elif 'keeshond' in img_name:
            animal = 'dog'
        elif 'pinscher' in img_name:
            animal = 'dog'
        elif 'leonberger' in img_name:
            animal = 'dog'
        elif 'Siamese' in img_name:
            animal = 'cat'
        elif 'Bombay' in img_name:
            animal = 'cat'
        elif 'pomeranian' in img_name:
            animal = 'dog'
        elif 'japanese' in img_name:
            animal = 'dog'
        elif 'beagle' in img_name:
            animal = 'dog'
        elif 'Birman' in img_name:
            animal = 'cat'
        elif 'Sphynx' in img_name:
            animal = 'cat'
        return animal


