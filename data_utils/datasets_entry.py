import os
import scipy.io as io
import numpy as np
from sklearn.model_selection import train_test_split
import data_utils.datasets_imu, data_utils.datasets_mmfi
import dill
from data_utils.pamap_util import gen_PAMAP_data, PAMAP_dict, gen_PAMAP_data2
from data_utils.base import z_score_standard_single
from data_utils.mmwave_util import gen_mmwave, load_act_label
from data_utils.wifi_csi_util import gen_wifi, gen_wifi2
from data_utils.widar_util import gen_widar_processed, load_widar_act_label, gen_widar_raw
from data_utils.lidar_utils import gen_lidar
from data_utils.ut_har_util import UT_HAR_dataset, load_act_label_ut_har
from data_utils.usc_had_util import gen_USC_data

def window(data, size, stride):
    '''
    save the data array into a list of arrays with the size of size and stride of stride
    '''

    x = []
    for i in range(0, data.shape[0], stride):
        if i+size <= data.shape[0]:
                x.append(data[i: i + size])
    return x

def merge(path, size, stride):
    '''
    combine the data from all subjects
    '''
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


def meta_generate(configs):
    if configs["dataset_args"]["pre_saved"]:
        log_dir = configs["dataset_args"]["pre_saved"]
    else:
        log_dir = configs["model_path"]

    if configs["dataset_args"]["dataset"] == "pamap":
        # if data not exist
        if not os.path.exists("data_cache/pamap_data.npy"):
            # gen_PAMAP_data2(dataset_dir=configs["dataset_args"]["dataset_path"], window_size=80, step=20)

            gen_PAMAP_data(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=171, OVERLAP_RATE=0.1, SPLIT_RATE=(8, 2),
                           VALIDATION_SUBJECTS={105}, Z_SCORE=True)

        all_data = np.load("data_cache/pamap_data.npy")
        all_label = np.load("data_cache/pamap_label.npy")
        action_idx_dict = PAMAP_dict()

        PAMAP_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        PAMAP_process.bind_label_text_dict(action_idx_dict)
        PAMAP_process.optimize_text(prefix=None, suffix=None)

        if configs["baseline_args"]["baseline"] == "sup":
            PAMAP_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            PAMAP_process.meta_building_no_unknown(seen_num=seen_num)


    if configs["dataset_args"]["dataset"] == "USC":
        if not os.path.exists(f'data_cache/USC_data.npy'):
            # result = merge(configs["dataset_args"]["dataset_path"], 512, 256)
            gen_USC_data(configs, window_size=128, stride=64)

        all_data = np.load("data_cache/USC_data.npy")
        all_label = np.load("data_cache/USC_label.npy")

        action_idx_dict = data_utils.datasets_imu.USC_HAD_dict()
        print("All classes of USC_HAD: ", action_idx_dict)
        print("Total class number: ", len(action_idx_dict))
        USC_HAD_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        USC_HAD_process.bind_label_text_dict(action_idx_dict)
        USC_HAD_process.optimize_text(prefix=None, suffix=None)

        if configs["baseline_args"]["baseline"] == "sup":
            USC_HAD_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            USC_HAD_process.meta_building_no_unknown(seen_num=seen_num)

    elif configs["dataset_args"]["dataset"] == "mmwave":
        ########## new implementation for mmwave ##########

        if not os.path.exists("data_cache/mmwave_data.npy"):
            gen_mmwave(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=100, OVERLAP_RATE=0.1)
            # gen_mmwave(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=512, OVERLAP_RATE=0.5)

        all_data = np.load("data_cache/mmwave_data.npy")
        all_label = np.load("data_cache/mmwave_label.npy")
        action_idx_dict = load_act_label()
        # mmfi_meta_builder = data_utils.datasets_mmfi.general_meta_build(configs, configs["dataset_args"]["dataset"], log_dir)

        mmwave_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        mmwave_process.bind_label_text_dict(action_idx_dict)
        mmwave_process.optimize_text(prefix=None, suffix=None)

        if configs["baseline_args"]["baseline"] == "sup":
            mmwave_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            mmwave_process.meta_building_no_unknown(seen_num=seen_num)
    elif configs["dataset_args"]["dataset"] == "wifi":
        if not os.path.exists("data_cache/wifi_data.npy"):
            # gen_wifi(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=40, OVERLAP_RATE=0.1)
            # gen_mmwave(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=512, OVERLAP_RATE=0.5)
            gen_wifi2(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=60, OVERLAP_RATE=0.1)

        all_data = np.load("data_cache/wifi_data.npy")
        all_label = np.load("data_cache/wifi_label.npy")
        action_idx_dict = load_act_label()
        # mmfi_meta_builder = data_utils.datasets_mmfi.general_meta_build(configs, configs["dataset_args"]["dataset"], log_dir)

        mmwave_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        mmwave_process.bind_label_text_dict(action_idx_dict)
        mmwave_process.optimize_text(prefix=None, suffix=None)

        if configs["baseline_args"]["baseline"] == "sup":
            mmwave_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            mmwave_process.meta_building_no_unknown(seen_num=seen_num)
    elif configs["dataset_args"]["dataset"] == "widar":
        if not os.path.exists("data_cache/widar_data.npy"):
            # gen_widar_processed(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=1024, OVERLAP_RATE=0.1)
            gen_widar_raw(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=128, OVERLAP_RATE=0.1)

        all_data = np.load("data_cache/widar_data.npy")
        all_label = np.load("data_cache/widar_label.npy")
        action_idx_dict = load_widar_act_label()

        widar_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        widar_process.bind_label_text_dict(action_idx_dict)
        widar_process.optimize_text(prefix=None, suffix=None)
        if configs["baseline_args"]["baseline"] == "sup":
            widar_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            widar_process.meta_building_no_unknown(seen_num=seen_num)
    elif configs["dataset_args"]["dataset"] == "lidar":
        if not os.path.exists("data_cache/lidar_data.npy"):
            # gen_widar_processed(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=1024, OVERLAP_RATE=0.1)
            gen_lidar(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=2048, OVERLAP_RATE=0.1)

        all_data = np.load("data_cache/lidar_data.npy")
        all_label = np.load("data_cache/lidar_label.npy")
        action_idx_dict = load_widar_act_label()

        lidar_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        lidar_process.bind_label_text_dict(action_idx_dict)
        lidar_process.optimize_text(prefix=None, suffix=None)
        if configs["baseline_args"]["baseline"] == "sup":
            lidar_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            lidar_process.meta_building_no_unknown(seen_num=seen_num)
    elif configs["dataset_args"]["dataset"] == "uthar":
        if not os.path.exists("data_cache/uthar_data.npy"):
            # gen_widar_processed(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=1024, OVERLAP_RATE=0.1)
            UT_HAR_dataset(dataset_dir=configs["dataset_args"]["dataset_path"], WINDOW_SIZE=50, OVERLAP_RATE=0.1)

        all_data = np.load("data_cache/uthar_data.npy")
        all_label = np.load("data_cache/uthar_label.npy")
        action_idx_dict = load_act_label_ut_har()

        uthar_process = data_utils.datasets_imu.general_meta_build(configs, all_data, all_label, configs["dataset_args"]["dataset"], log_dir)
        uthar_process.bind_label_text_dict(action_idx_dict)
        uthar_process.optimize_text(prefix=None, suffix=None)
        if configs["baseline_args"]["baseline"] == "sup":
            uthar_process.meta_supervised()
        else:
            seen_num = int(configs["dataset_args"]["seen_num"])
            uthar_process.meta_building_no_unknown(seen_num=seen_num)


if __name__ == '__main__':
    pass