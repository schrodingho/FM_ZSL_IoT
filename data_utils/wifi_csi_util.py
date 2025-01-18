import scipy.io as scio
import os
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

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

def gen_wifi(dataset_dir='/your_path', WINDOW_SIZE=1024, OVERLAP_RATE=0.1):

    all_x = []
    all_y = []
    length = []
    subject_list = os.listdir(dataset_dir)
    wifi_csi_path = "wifi-csi"
    print('###################### Loading wifi-csi data ######################')
    subj_cnt = 0
    for subject in subject_list:
        # reduce the number of data
        subj_cnt += 1
        if subj_cnt > 5:
            break
        action_list = os.listdir(os.path.join(dataset_dir, subject))
        for action in action_list:
            sub_path = os.path.join(dataset_dir, subject, action, wifi_csi_path)
            action_id = int(action[1:]) - 1
            mat_list = os.listdir(sub_path)
            cur_seq = []
            for mat in mat_list:
                frame_path = os.path.join(sub_path, mat)
                data = scio.loadmat(frame_path)['CSIamp']

                data[np.isinf(scio.loadmat(frame_path)['CSIamp'])] = np.nan
                for i in range(10):  # 32
                    temp_col = data[:, :, i]
                    nan_num = np.count_nonzero(temp_col != temp_col)
                    if nan_num != 0:
                        temp_not_nan_col = temp_col[temp_col == temp_col]
                        temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
                    data[:, :, i] = temp_col
                first_dim_shape = data.shape[0]
                second_dim_shape = data.shape[1]
                last_dim_shape = data.shape[2]
                data = data.reshape(first_dim_shape * second_dim_shape, last_dim_shape)
                data = data.T
                cur_seq.append(data)

            cur_data = np.concatenate(cur_seq, axis=0)
            cur_data = sliding_window(cur_data, WINDOW_SIZE, OVERLAP_RATE)
            all_x += cur_data
            all_y += [action_id] * len(cur_data)

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_x = z_score_standard_single(all_x)

    # print(all_x.shape)
    np.save('./data_cache/wifi_data.npy', all_x)
    np.save('./data_cache/wifi_label.npy', all_y)

def gen_wifi2(dataset_dir='/your_path', WINDOW_SIZE=1024, OVERLAP_RATE=0.1):
    all_x = []
    all_y = []
    length = []
    subject_list = os.listdir(dataset_dir)
    wifi_csi_path = "wifi-csi"
    print('###################### Loading wifi-csi data ######################')
    subj_cnt = 0
    # selected_subcarriers = list(range(50, 70))
    # all_subcarriers_idx = np.arange(114)
    # num_subcarriers = 20
    selected_subcarriers = list(range(25, 100))
    for subject in subject_list:
        subj_cnt += 1
        if subj_cnt > 6:
            break
        action_list = os.listdir(os.path.join(dataset_dir, subject))
        for action in action_list:
            sub_path = os.path.join(dataset_dir, subject, action, wifi_csi_path)
            action_id = int(action[1:]) - 1
            mat_list = os.listdir(sub_path)
            cur_seq = []
            for mat in mat_list:
                frame_path = os.path.join(sub_path, mat)
                data = scio.loadmat(frame_path)['CSIamp']
                data[np.isinf(scio.loadmat(frame_path)['CSIamp'])] = 0
                # for i in range(10):  # 32
                #     temp_col = data[:, :, i]
                #     nan_num = np.count_nonzero(temp_col != temp_col)
                #     if nan_num != 0:
                #         temp_not_nan_col = temp_col[temp_col == temp_col]
                #         temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
                #     data[:, :, i] = temp_col
                data = data[:, selected_subcarriers, :]
                
                # data = select_pca(data, num_subcarriers=10)
                # data = downsampling_subcarrier_signal(data, window_size=2)
                first_dim_shape = data.shape[0]
                second_dim_shape = data.shape[1]
                last_dim_shape = data.shape[2]
                data = data.reshape(first_dim_shape * second_dim_shape, last_dim_shape)
                data = data.T
                cur_seq.append(data)

            cur_data = np.concatenate(cur_seq, axis=0)
            cur_data = sliding_window(cur_data, WINDOW_SIZE, OVERLAP_RATE)
            all_x += cur_data
            all_y += [action_id] * len(cur_data)

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_x = z_score_standard_single(all_x)

    # print(all_x.shape)
    np.save('./data_cache/wifi_data.npy', all_x)
    np.save('./data_cache/wifi_label.npy', all_y)


def select_important_subcarriers(csi_data, num_subcarriers=20):
    first_dim_shape = csi_data.shape[0]
    second_dim_shape = csi_data.shape[1]
    last_dim_shape = csi_data.shape[2]
    csi_2d = csi_data.reshape((first_dim_shape * second_dim_shape, last_dim_shape))
    # Apply TruncatedSVD
    svd = TruncatedSVD(n_components=num_subcarriers)
    selected_csi_2d = svd.fit_transform(csi_2d)
    selected_csi_data = selected_csi_2d.reshape((first_dim_shape, num_subcarriers, last_dim_shape))
    return selected_csi_data

def select_pca(csi_data, num_subcarriers=40):
    first_dim_shape = csi_data.shape[0]
    second_dim_shape = csi_data.shape[1]
    last_dim_shape = csi_data.shape[2]
    csi_2d = csi_data.reshape((first_dim_shape * second_dim_shape, last_dim_shape)).T
    # Apply PCA
    pca = PCA(n_components=num_subcarriers)
    selected_csi_2d = pca.fit_transform(csi_2d)

    return selected_csi_2d



def select_important_subcarriers_2(csi_data, num_subcarriers=20):
    # 计算每个子载波上的能量
    energy = np.sum(np.abs(csi_data) ** 2, axis=1)

    # 计算每个子载波上的方差
    # variance = np.var(csi_data, axis=1)

    # 综合考虑能量和方差，计算每个子载波的重要性得分
    importance_score = energy

    # 选择重要性得分最高的20个子载波
    important_subcarriers_idx = np.argsort(importance_score, axis=None)[-num_subcarriers:]

    return important_subcarriers_idx


def downsampling_subcarrier_signal(csi_data, window_size=5):
    sub_carrier_num = csi_data.shape[1]
    downsampled_sub_carrier_num = sub_carrier_num // window_size
    output_csi_data = np.zeros((csi_data.shape[0], downsampled_sub_carrier_num, csi_data.shape[2]))
    for i in range(downsampled_sub_carrier_num):
        csi_data[:, i, :] = np.mean(csi_data[:, i * window_size:(i + 1) * window_size, :], axis=1)
    
    return output_csi_data

if __name__ == '__main__':
    # gen_wifi()
    gen_wifi2()