import scipy.io as scio
import os
import numpy as np
import sys
import scipy.io

os.chdir(sys.path[0])
sys.path.append('../')
from data_utils.base import *
import pandas as pd

def load_widar_act_label():
    action_list = ['Push and Pull', 'Sweep', 'Clap', 'Slide', 'Draw N (Horizontal)', 'Draw Circle (Horizontal)',
     'Draw Rectangle (Horizontal)', 'Draw Triangle (Horizontal)', 'Draw Zigzag (Horizontal)', 'Draw Zigzag (Vertical)',
     'Draw N (Vertical)', 'Draw Circle (Vertical)', 'Draw Number 1', 'Draw Number 2', 'Draw Number 3',
     'Draw Number 4', 'Draw Number 5', 'Draw Number 6', 'Draw Number 7', 'Draw Number 8', 'Draw Number 9',
     'Draw Number 10']
    act_idx_desp_dict = {
        0: "Push and Pull",
        1: "Sweep",
        2: "Clap",
        3: "Slide",
        4: "Draw N (Horizontal)",
        5: "Draw Circle (Horizontal)",
        6: "Draw Rectangle (Horizontal)",
        7: "Draw Triangle (Horizontal)",
        8: "Draw Zigzag (Horizontal)",
        9: "Draw Zigzag (Vertical)",
        10: "Draw N (Vertical)",
        11: "Draw Circle (Vertical)",
        12: "Draw Number 1",
        13: "Draw Number 2",
        14: "Draw Number 3",
        15: "Draw Number 4",
        16: "Draw Number 5",
        17: "Draw Number 6",
        18: "Draw Number 7",
        19: "Draw Number 8",
        20: "Draw Number 9",
        21: "Draw Number 10",
    }
    action_small = {
        0: "Push and Pull",
        1: "Sweep",
        2: "Clap",
        3: "Slide",
        4: "Draw Circle",
        5: "Draw Zigzag",
    }
    small_list = ['Push and Pull', 'Sweep', 'Clap', 'Slide', 'Draw Circle', 'Draw Zigzag']



    action_map = {
        '1-Push&Pull': 0,
        '2-Sweep': 1,
        '3-Clap': 2,
        '4-Slide': 3,
        '6-Draw-O(H)': 4,
        '12-Draw-O(V)': 4,
        '9-Draw-Zigzag(H)': 5,
        '10-Draw-Zigzag(V)': 5,
    }

    return action_small

def list_all_dir_names(path="/home/dingding/PycharmProjects/WiFi-CSI-Sensing-Benchmark/Data/Widardata/test"):
    all_dirs = os.listdir(path)
    print(all_dirs)

def gen_widar_processed(dataset_dir='/home/dingding/PycharmProjects/WiFi-CSI-Sensing-Benchmark/Data/Widardata/test', WINDOW_SIZE=1024, OVERLAP_RATE=0.1):
    all_x = []
    all_y = []
    classes_list = os.listdir(dataset_dir)

    action_small = {
        0: "Push and Pull",
        1: "Sweep",
        2: "Clap",
        3: "Slide",
        4: "Draw Circle",
        5: "Draw Zigzag",
    }

    action_map = {
        '1-Push&Pull': 0,
        '2-Sweep': 1,
        '3-Clap': 2,
        '4-Slide': 3,
        '6-Draw-O(H)': 4,
        '12-Draw-O(V)': 4,
        '9-Draw-Zigzag(H)': 5,
        '10-Draw-Zigzag(V)': 5,
    }


    print('###################### Loading widar data ######################')
    for cur_class in classes_list:
        file_list = os.listdir(os.path.join(dataset_dir, cur_class))
        if cur_class not in action_map:
            continue
        class_id = action_map[cur_class]
        # class_id = int(cur_class.split("-")[0]) - 1
        # if class_id > 10:
        #     continue
        cur_seq = []
        for cur_file in file_list:
            cur_path = os.path.join(dataset_dir, cur_class, cur_file)
            x = np.genfromtxt(cur_path, delimiter=',')
            x = (x - 0.0025) / 0.0119
            x = x.T
            x = np.expand_dims(x, axis=0)
            cur_seq.append(x)
        cur_data = np.concatenate(cur_seq, axis=0)
        # cur_data = sliding_window(cur_data, WINDOW_SIZE, OVERLAP_RATE)
        all_x += cur_data.tolist()
        all_y += [class_id] * len(cur_data)

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_x = z_score_standard_single(all_x)
    # print(all_x.shape)
    # print(all_y.shape)


    np.save('./data_cache/widar_data.npy', all_x)
    np.save('./data_cache/widar_label.npy', all_y)


def gen_widar_raw(dataset_dir='/home/dingding/Datasets/widar_3_0/BVP', WINDOW_SIZE=128, OVERLAP_RATE=0.1):
    src_path = dataset_dir
    link_path = "6-link"

    selected_action = ['Push and Pull', 'Sweep', 'Clap', 'Slide', 'Draw Circle', 'Draw Zigzag']

    all_dir = os.listdir(src_path)

    seq_dict = {}

    for sub_dir in all_dir:
        if widar_raw_all_dict[sub_dir] == {}:
            continue
        print(sub_dir)
        if sub_dir == "20181130-VS":
            sub_dir_path = os.path.join(src_path, sub_dir)
        else:
            sub_dir_path = os.path.join(src_path, sub_dir, link_path)
        # list all files in sub_dir
        all_users = os.listdir(sub_dir_path)
        for user in all_users:
            cur_user_path = os.path.join(sub_dir_path, user)
            all_files = os.listdir(cur_user_path)
            for cur_file in all_files:
                class_id = int(cur_file.split("-")[1])
                class_name = widar_raw_all_dict[sub_dir][class_id]
                if class_name not in seq_dict:
                    seq_dict[class_name] = []
                cur_file_path = os.path.join(cur_user_path, cur_file)
                file_size = os.path.getsize(cur_file_path)

                if file_size == 0:
                    print(f"File {cur_file_path} appears to be empty.")
                    continue
                mat = scipy.io.loadmat(cur_file_path)
                cur_data = mat["velocity_spectrum_ro"]
                last_dim_shape = cur_data.shape[-1]
                if last_dim_shape == 0:
                    continue
                cur_data = cur_data.reshape(-1, last_dim_shape).T
                new_last_dim_shape = cur_data.shape[-1]
                # interpolate to 400?
                if new_last_dim_shape != 400:
                    continue

                # cur_data = cur_data.interpolate(method='linear', limit_direction='forward',
                #                               axis=0).to_numpy()  # 线性插值填充nan
                seq_dict[class_name].append(cur_data)

                # print(cur_data.shape)
    print(seq_dict.keys())
    # selected_action = ['Push and Pull', 'Sweep', 'Clap', 'Slide', 'Draw Circle', 'Draw Zigzag']
    all_actions = ['Push and Pull', 'Draw Zigzag (Vertical)', 'Clap', 'Draw Circle (Vertical)', 'Draw N (Vertical)',
                   'Sweep', 'Draw Zigzag (Horizontal)', 'Slide', 'Draw Circle (Horizontal)',
                   'Draw Triangle (Horizontal)', 'Draw N (Horizontal)', 'Draw Rectangle (Horizontal)']

    action_small = {
        0: "Push and Pull",
        1: "Sweep",
        2: "Clap",
        3: "Slide",
        4: "Draw Circle",
        5: "Draw Zigzag",
    }

    map_to_small = {
        'Push and Pull': 0,
        'Sweep': 1,
        'Clap': 2,
        'Slide': 3,
        'Draw Circle (Vertical)': 4,
        'Draw Circle (Horizontal)': 4,
        'Draw Zigzag (Vertical)': 5,
        'Draw Zigzag (Horizontal)': 5,
    }

    all_x = []
    all_y = []

    for key, val in seq_dict.items():
        if key not in map_to_small:
            continue
        print(f"Processing {key}")
        class_id = map_to_small[key]
        cur_seq = val
        data_cur = np.concatenate(cur_seq, axis=0)
        data_cur = sliding_window(data_cur, WINDOW_SIZE, OVERLAP_RATE)
        all_x += data_cur
        all_y += [class_id] * len(data_cur)

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_x = z_score_standard_single(all_x)
    print(all_x.shape)
    print(all_y.shape)
    np.save('./data_cache/widar_data.npy', all_x)
    np.save('./data_cache/widar_label.npy', all_y)

if __name__ == '__main__':
    # gen_widar_processed()
    list_all_dir_names()