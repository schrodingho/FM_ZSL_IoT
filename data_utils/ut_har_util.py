import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from data_utils.base import *
def load_act_label_ut_har():
    act_idx_word_dict = {
        0: "Lie down",
        1: "Fall",
        2: "Walk",
        3: "Pick up",
        4: "Run",
        5: "Sit down",
        6: "Stand up",
    }

    return act_idx_word_dict

def UT_HAR_dataset(dataset_dir="/your_path", WINDOW_SIZE=50, OVERLAP_RATE=0.1):
    data_list = glob.glob(dataset_dir+'/data/*.csv')
    label_list = glob.glob(dataset_dir+'/label/*.csv')
    all_x = []
    all_y = []
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            # raw data shape (496, 250, 90)

            # cur_data = sliding_window(cur_data, WINDOW_SIZE, OVERLAP_RATE)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            all_x.append(data_norm)

    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
            label = label.tolist()
            all_y += label

    all_x = np.concatenate(all_x, axis=0)
    all_x, all_y = np.array(all_x), np.array(all_y)


    print(f"UT_HAR dataset shape: {all_x.shape}, {all_y.shape}")

    np.save('./data_cache/uthar_data.npy', all_x)
    np.save('./data_cache/uthar_label.npy', all_y)




if __name__ == '__main__':
    UT_HAR_dataset()