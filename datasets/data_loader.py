import os.path

import torch

# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
import random
# import util.util as util
from torchvision import transforms
import SimpleITK as sitk
from monai import transforms, data
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
    '.nii', '.nii.gz', '.dcm', '.dicom',
    '.NII', '.NII.GZ', '.DCM', '.DICOM',
    '.npy', '.npz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class MMWHSDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, data_path, gt_path, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.data_path = data_path
        self.gt_path = gt_path

        self.A_paths = sorted(make_dataset(data_path))
        self.B_paths = sorted(make_dataset(gt_path))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert self.A_size == self.B_size

        self.classes = ['BG', 'Myo', 'LAC', 'LVC', 'AA']
        self.transform = self.get_transform(phase=phase)

        # ID_TO_TRAIN_ID = {
        #     0: 255, 1: 0, 2: 1, 3: 2, 4: 3
        # }
        # TRAIN_ID_TO_COLOR = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        #                      (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        #                      (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        #                      (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
        #                      (0, 0, 230), (119, 11, 32), [0, 0, 0]]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = os.path.join(self.gt_path, os.path.basename(A_path).replace('.npy', '_gt.npy'))

        data = self.transform({'image': A_path, 'label': B_path})

        # print(data['image'].shape, data['label'].shape)
        # (3, 256, 256) (1, 256, 256)

        return data['image'], data['label'].squeeze(0)

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_transform(self, keys=('image', 'label'), phase='train'):
        # mode = ['bilinear', 'nearest']

        if phase == 'train':
            transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=keys, reader='NumpyReader'),
                    transforms.EnsureChannelFirstd(keys=keys),
                    transforms.RepeatChanneld(keys=keys[:-1], repeats=3),
                    transforms.ToTensord(keys=keys),
                ]
            )
        elif phase == 'val':
            transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=keys, reader='NumpyReader'),
                    transforms.EnsureChannelFirstd(keys=keys),
                    transforms.RepeatChanneld(keys=keys[:-1], repeats=3),
                    transforms.ToTensord(keys=keys),
                ]
            )

        elif phase == 'test':
            transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=keys, reader='NumpyReader'),
                    transforms.EnsureChannelFirstd(keys=keys),
                    transforms.RepeatChanneld(keys=keys[:-1], repeats=3),
                    transforms.ToTensord(keys=keys),
                ]
            )
        else:
            transform = None

        return transform

    @property
    def evaluate_classes(self):
        return [
            'Myo', 'LAC', 'LVC', 'AA'
        ]

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    def decode_target(self, target):
        """ Decode label (each value is integer) into the corresponding RGB value.

        Args:
            target (numpy.array): label in shape H x W

        Returns:
            RGB label (PIL Image) in shape H x W x 3
        """
        target = target.copy()
        target[target == 255] = self.num_classes # unknown label is black on the RGB label
        target = self.train_id_to_color[target]
        return Image.fromarray(target.astype(np.uint8))

    def collect_image_paths(self):
        """Return a list of the absolute path of all the images"""
        return [os.path.join(self.root, self.data_folder, image_name) for image_name in self.data_list]


def get_transform(opt, keys=['image'], label=False):
    MIN = 0.
    MAX = 1.
    mode = ['bilinear', 'nearest']
    from monai.data.image_reader import NumpyReader
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CropForegroundd(keys=keys, source_key="image"),
            # transforms.SpatialPadd(keys=keys, spatial_size=(196, 196, -1), mode='reflect'),
            # CheckDim(keys=keys),
            # transforms.RandSpatialCropd(keys=keys,
            #                             roi_size=(176, 176, 1),
            #                             random_size=False),
            # CheckDim(keys=keys),

            # transforms.Resized(keys=keys, spatial_size=(256, 256, 1),
            #                    mode=mode),
            # CheckDim(keys=keys),

            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            # transforms.Rand3DElasticd(keys=keys, sigma_range=(5, 7), magnitude_range=(50, 150),
            #                           padding_mode='zeros', mode=mode),
            # transforms.RandCropByPosNegLabeld(
            #     keys=keys,
            #     label_key="image",
            #     spatial_size=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=0),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=1),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=2),
            # transforms.RandRotate90d(
            #     keys=keys,
            #     prob=0.2,
            #     max_k=3,
            # ),
            # transforms.RandScaleIntensityd(keys=keys[:-1], factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys=keys[:-1], offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=keys),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CenterSpatialCropd(keys=keys, roi_size=(176, 176, 1), ),
            # transforms.Resized(keys=keys, spatial_size=(256, 256, -1), mode=mode),
            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            transforms.ToTensord(keys=keys),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CenterSpatialCropd(keys=keys, roi_size=(176, 176, -1), ),
            # transforms.Resized(keys=keys, spatial_size=(256, 256, -1), mode=mode),
            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            transforms.ToTensord(keys=keys),
        ]
    )

    if opt.isTrain and opt.phase == 'train':
        return train_transform
    elif opt.isTrain and opt.phase == 'test':
        return val_transform
    else:
        return test_transform


from monai.transforms.transform import MapTransform
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor


class ScaleMinMaxNorm(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = ((d[key] - d[key].min()) / (d[key].max() - d[key].min())) * (self.a_max - self.a_min) + self.a_min
        return d


class DeNorm(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key] + 1) * 127.5
        return d


class CheckDim(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            print(key, d[key].shape)
        return d
