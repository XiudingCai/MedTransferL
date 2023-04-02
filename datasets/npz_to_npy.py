import os

import numpy as np
from glob import glob
from tqdm import tqdm
import shutil


def npz_to_npy():
    TARGET_MODALITY = 'CT'
    ct_path = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/test_ct"
    save_path = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/test_ct"
    save_path_gt = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_test_ct"
    os.makedirs(save_path_gt, exist_ok=True)
    slice_count = 0
    for path in tqdm(glob(os.path.join(ct_path, "*.npz"))):

        _npz_dict = np.load(path)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1'].astype(int)

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        if TARGET_MODALITY == 'CT':
            data = np.subtract(
                np.multiply(np.divide(np.subtract(data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
        elif TARGET_MODALITY == 'MR':
            data = np.subtract(
                np.multiply(np.divide(np.subtract(data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
        print(data.min(), data.max())

        for z in range(data.shape[2]):
            item_data = data[:, :, z]
            item_label = label[:, :, z]

            np.save(f"{save_path}/ct_test_slice{slice_count}.npy", item_data)
            np.save(f"{save_path_gt}/ct_test_slice{slice_count}_gt.npy", item_label)
            slice_count += 1

    TARGET_MODALITY = 'MR'
    mr_path = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/test_mr"
    save_path = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/test_mr"
    save_path_gt = "/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_test_mr"
    os.makedirs(save_path_gt, exist_ok=True)
    slice_count = 0

    for path in tqdm(glob(os.path.join(mr_path, "*.npz"))):

        _npz_dict = np.load(path)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1']

        if TARGET_MODALITY == 'CT':
            data = np.subtract(
                np.multiply(np.divide(np.subtract(data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
        elif TARGET_MODALITY == 'MR':
            data = np.subtract(
                np.multiply(np.divide(np.subtract(data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
        print(data.min(), data.max())

        for z in range(data.shape[2]):
            item_data = data[:, :, z]
            item_label = label[:, :, z]

            np.save(f"{save_path}/mr_test_slice{slice_count}.npy", item_data)
            np.save(f"{save_path_gt}/mr_test_slice{slice_count}_gt.npy", item_label)
            slice_count += 1


def split_datasets():
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/train_ct",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/trainval_ct", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_train_ct",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_trainval_ct", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/val_ct",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/trainval_ct", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_val_ct",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_trainval_ct", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/train_mr",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/trainval_mr", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_train_mr",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_trainval_mr", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/val_mr",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/trainval_mr", dirs_exist_ok=True)
    shutil.copytree(src="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_val_mr",
                    dst="/home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np/gt_trainval_mr", dirs_exist_ok=True)

npz_to_npy()
# split_datasets()
