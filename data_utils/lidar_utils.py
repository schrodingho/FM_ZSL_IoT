import scipy.io as scio
import os
import numpy as np
import sys

os.chdir(sys.path[0])
sys.path.append('../')
from data_utils.base import *
import pandas as pd
def load_act_label():
    act_label_path = "./data_cache/MMFi_act.csv"
    act_df = pd.read_csv(act_label_path, sep="\t")
    # print(act_label)
    # Build the dictionary between activity and description
    act_code_desp_dict = act_df.set_index('Activity')['Description'].to_dict()
    act_idx_desp_dict = {}
    for act, desp in act_code_desp_dict.items():
        action_id = int(act[1:]) - 1
        act_idx_desp_dict[action_id] = desp

    return act_idx_desp_dict

def gen_lidar(dataset_dir='/home/dingding/Datasets/lidar_data/select_lidar/E04', WINDOW_SIZE=2048, OVERLAP_RATE=0.1):

    all_x = []
    all_y = []
    length = []
    subject_list = os.listdir(dataset_dir)
    wifi_csi_path = "lidar"
    print('###################### Loading lidar data ######################')
    for subject in subject_list:
        action_list = os.listdir(os.path.join(dataset_dir, subject))
        for action in action_list:
            sub_path = os.path.join(dataset_dir, subject, action, wifi_csi_path)
            action_id = int(action[1:]) - 1
            mat_list = os.listdir(sub_path)
            cur_seq = []
            for mat in mat_list:
                frame_path = os.path.join(sub_path, mat)
                with open(frame_path, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 3)
                    cur_seq.append(data_tmp)
            cur_data = np.concatenate(cur_seq, axis=0)
            cur_data = sliding_window(cur_data, WINDOW_SIZE, OVERLAP_RATE)
            all_x += cur_data
            all_y += [action_id] * len(cur_data)

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_x = z_score_standard_single(all_x)

    # print(all_x.shape)
    np.save('./data_cache/lidar_data.npy', all_x)
    np.save('./data_cache/lidar_label.npy', all_y)


def gen_mmwave_method_2():
    return

if __name__ == '__main__':
    gen_lidar()