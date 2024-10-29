import dill
import numpy as np
import torch
import os

def extract_raw_func(config, trn_meta, tst_seen_meta, tst_unseen_meta, seen_text_file, unseen_text_file, all_data, type="seen"):
    dataset_name = config["dataset_args"]["dataset"]
    unseen_num = config["dataset_args"]["unseen_num"]
    save_path = config["extract_raw_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_action_dict = read_text_file(seen_text_file, unseen_text_file)
    dill.dump(all_action_dict, open(save_path + "/all_text_label_dict.pkl", "wb"))
    ###################################################################################################################

    trn_x_all, trn_y_all, trn_unique_y = USC_PAMAP_extraction(trn_meta, all_data)
    tst_x_all, tst_y_all, tst_unique_y = USC_PAMAP_extraction(tst_seen_meta, all_data)
    tst_x_all_unseen, tst_y_all_unseen, tst_unique_y_unseen = USC_PAMAP_extraction(tst_unseen_meta, all_data)

    trn_select_targets = trn_unique_y[:len(trn_unique_y) - unseen_num]
    val_select_targets = trn_unique_y[len(trn_unique_y) - unseen_num:]

    torch.save(trn_x_all, save_path + "/trn_feat.pth")
    torch.save(trn_y_all, save_path + "/trn_targets.pth")
    torch.save(trn_unique_y, save_path + "/seen_targets.pth")
    torch.save(trn_select_targets, save_path + "/trn_select_targets.pth")
    torch.save(val_select_targets, save_path + "/val_select_targets.pth")

    ###################################################################################################################

    tst_y_all_unseen += len(trn_unique_y)
    tst_unique_y_unseen += len(trn_unique_y)


    torch.save(tst_x_all, save_path + "/test_seen_feat.pth")
    torch.save(tst_y_all, save_path + "/test_seen_targets.pth")

    torch.save(tst_x_all_unseen, save_path + "/test_unseen_feat.pth")
    torch.save(tst_y_all_unseen, save_path + "/test_unseen_targets.pth")

    torch.save(tst_unique_y_unseen, save_path + "/unseen_targets.pth")

    print("Raw Data for augmentation has been saved!")

def USC_PAMAP_extraction(meta, all_data):
    meta_data_list = meta["data_list"]
    x_list = []
    y_list = []
    y_set = set()

    for item in meta_data_list:
        x = all_data[item["file"]]
        x = x.reshape(1, -1)
        y = item["label"]

        x_list.append(x)
        y_list.append(y)
        y_set.add(y)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.array(y_list)
    unique_y = np.array(list(y_set))
    unique_y = np.sort(unique_y)

    x_all = torch.from_numpy(x_all).float()
    y_all = torch.from_numpy(y_all).long()
    unique_y = torch.from_numpy(unique_y).long()

    return x_all, y_all, unique_y

def PAMAP_extraction(meta):
    meta_data_list = meta["data_list"]
    x_list = []
    y_list = []
    y_set = set()

    for item in meta_data_list:
        x = all_data[item["file"]]
        x = x.reshape(1, -1)
        y = item["label"]

        x_list.append(x)
        y_list.append(y)
        y_set.add(y)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.array(y_list)
    unique_y = np.array(list(y_set))
    unique_y = np.sort(unique_y)

    x_all = torch.from_numpy(x_all).float()
    y_all = torch.from_numpy(y_all).long()
    unique_y = torch.from_numpy(unique_y).long()

    return x_all, y_all, unique_y

def read_text_file(seen_file_path, unseen_file_path):
    with open(seen_file_path, "rb") as f:
        seen_lines = f.readlines()
    seen_action_list = [a.decode('utf-8').split('\n')[0] for a in seen_lines]

    with open(unseen_file_path, "rb") as f:
        unseen_lines = f.readlines()
    unseen_action_list = [a.decode('utf-8').split('\n')[0] for a in unseen_lines]

    all_action_list = seen_action_list + unseen_action_list
    all_action_dict = {all_action_list[i]: i for i in range(len(all_action_list))}
    return all_action_dict


if __name__ == '__main__':
    meta_path = "/home/dingding/PycharmProjects/Efficient-Prompt/logs_USC/logdir_20240408-115741"

    config = {
        "dataset_args": {
            "dataset": "USC",
            "unseen_num": 3
        },
        "extract_raw_dir": meta_path + "/extracted_raw"

    }
    dataset_name = config["dataset_args"]["dataset"]
    if config["dataset_args"]["dataset"] == "USC":
        tmp_path = "./data_cache/all_data.pkl"
        all_data = dill.load(open(tmp_path, "rb"))
    elif config["dataset_args"]["dataset"] == "pamap":
        tmp_path = "./data_cache/pamap2_data.npy"
        all_data = np.load(tmp_path)
    train_meta = dill.load(open(meta_path + "/train_meta.pkl", "rb"))
    tst_seen_meta = dill.load(open(meta_path + "/val_seen_meta.pkl", "rb"))
    tst_unseen_meta = dill.load(open(meta_path + "/val_unseen_meta.pkl", "rb"))
    seen_text_path = meta_path + f"/{dataset_name}_train.txt"
    unseen_text_path = meta_path + f"/{dataset_name}_val.txt"


    extract_raw_func(config, train_meta, tst_seen_meta, tst_unseen_meta, seen_text_path, unseen_text_path, all_data)
    # read_text_file(seen_text_path, unseen_text_path)