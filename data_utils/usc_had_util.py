import os
import numpy as np
import scipy.io as io
from data_utils.base import z_score_standard_single
import dill

def window(data, size, stride):
    x = []
    for i in range(0, data.shape[0], stride):
        if i+size <= data.shape[0]:
                x.append(data[i: i + size])
    return x

def merge(path, size, stride):
    result = [[] for i in range(12)]

    # list all folders
    subject_list = os.listdir(path)
    for subject in subject_list:
        # list all files in the subject folder
        mat_list = os.listdir(os.path.join(path, subject))
        for mat in mat_list:
            category = int(mat[1:-6])-1
            content = io.loadmat(path + subject + "/" + mat)['sensor_readings']
            x = window(content, size, stride)
            result[category].extend(x)

    return result


# gen_PAMAP_data(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=171, OVERLAP_RATE=0.1,
#                SPLIT_RATE=(8, 2),
#                VALIDATION_SUBJECTS={105}, Z_SCORE=True)

def gen_USC_data(configs, window_size, stride):
    result = merge(configs["dataset_args"]["dataset_path"], window_size, stride)
    all_data = result[0]
    all_y_true = np.array([0 for i in range(len(result[0]))])
    for i, data in enumerate(result):
        if i == 0:
            continue
        else:
            all_data = np.concatenate((all_data, data), axis=0)
            all_y_true = np.concatenate((all_y_true, np.array([i for n in range(len(data))])), axis=0)
    all_data = z_score_standard_single(all_data)
    np.save('./data_cache/USC_data.npy', all_data)
    np.save('./data_cache/USC_label.npy', all_y_true)
    # dill.dump(all_data, open(f'data_cache/all_data.pkl', 'wb'))
    # dill.dump(all_y_true, open(f'data_cache/all_y_true.pkl', 'wb'))