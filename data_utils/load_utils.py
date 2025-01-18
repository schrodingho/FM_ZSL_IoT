import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import dill
import scipy.io as scio
from data_utils.base import static_unknown_text

def setup_imu_data_loader(config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data, fake_dict=None):
    train_dataset = USC_Dataset(config, root_dir, train_meta, all_data)
    val_tune_dataset = USC_Dataset(config, root_dir, val_tune_meta, all_data)
    val_mix_dataset = USC_Dataset(config, root_dir, val_mix_meta, all_data)
    if fake_dict:
        real_number = len(train_meta["data_list"])
        fake_dict["real_number"] = real_number
        fake_dataset = USC_Dataset_fake(config, root_dir, train_meta, all_data, fake_dict)
        return [train_dataset, val_tune_dataset, val_mix_dataset, fake_dataset]
    return [train_dataset, val_tune_dataset, val_mix_dataset]

def setup_pamap_data_loader(config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data, fake_dict=None):
    train_dataset = PAMAP_Dataset(config, root_dir, train_meta, all_data)
    val_tune_dataset = PAMAP_Dataset(config, root_dir, val_tune_meta, all_data)
    val_mix_dataset = PAMAP_Dataset(config, root_dir, val_mix_meta, all_data)
    if fake_dict:
        real_number = len(train_meta["data_list"])
        fake_dict["real_number"] = real_number
        fake_dataset = PAMAP_Dataset_fake(config, root_dir, train_meta, all_data, fake_dict)
        return [train_dataset, val_tune_dataset, val_mix_dataset, fake_dataset]
    return [train_dataset, val_tune_dataset, val_mix_dataset]

def setup_mmwave_new_data_loader(config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data, fake_dict=None):
    train_dataset = mmwave_new_Dataset(config, root_dir, train_meta, all_data)
    val_tune_dataset = mmwave_new_Dataset(config, root_dir, val_tune_meta, all_data)
    val_mix_dataset = mmwave_new_Dataset(config, root_dir, val_mix_meta, all_data)
    if fake_dict:
        real_number = len(train_meta["data_list"])
        fake_dict["real_number"] = real_number
        fake_dataset = mmwave_new_Dataset_fake(config, root_dir, train_meta, all_data, fake_dict)
        return [train_dataset, val_tune_dataset, val_mix_dataset, fake_dataset]
    return [train_dataset, val_tune_dataset, val_mix_dataset]


class USC_Dataset(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data):
        self.root_dir = root_dir
        self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config

    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.data_list[idx]["label"]

        if self.config["dataset_args"]["original_text"] == True:
            out_text = self.data_list[idx]["text"]
        else:
            out_text = self.data_list[idx]["opt_text"]

        data_idx = self.data_list[idx]["file"]
        x = self.all_data[data_idx]

        vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type


class USC_Dataset_fake(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data, fake_dict):
        self.root_dir = root_dir

        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config
        self.fake_feat = torch.from_numpy(fake_dict["fake_feat"]).float().unsqueeze(1)
        if config["dataset_args"]["unknown"] == True:
            self.fake_targets = [int(config["dataset_args"]["seen_num"]) for _ in range(len(fake_dict["fake_targets"]))]
            self.fake_text = [static_unknown_text for _ in range(len(fake_dict["fake_targets"]))]
        else:
            self.fake_targets = fake_dict["fake_targets"]
            self.fake_text = fake_dict["fake_text"]
        self.fake_vis = np.ones((len(self.fake_targets),))
        self.real_number = fake_dict["real_number"]
        self.meta_data_len = fake_dict["real_number"] + len(self.fake_targets)


    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= self.real_number:
            idx = idx - self.real_number
            y = self.fake_targets[idx]
            out_text = self.fake_text[idx]
            x = self.fake_feat[idx]
            vis_type = self.fake_vis[idx]
        else:
            y = self.data_list[idx]["label"]

            if self.config["dataset_args"]["original_text"] == True:
                out_text = self.data_list[idx]["text"]
            else:
                out_text = self.data_list[idx]["opt_text"]

            data_idx = self.data_list[idx]["file"]
            x = self.all_data[data_idx]

            vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type

class PAMAP_Dataset(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data):
        self.root_dir = root_dir
        self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config

    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.data_list[idx]["label"]

        if self.config["dataset_args"]["original_text"] == True:
            out_text = self.data_list[idx]["text"]
        else:
            out_text = self.data_list[idx]["opt_text"]

        data_idx = self.data_list[idx]["file"]
        # shape (1, 171, 36)
        x = self.all_data[data_idx]

        vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type

class PAMAP_Dataset_fake(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data, fake_dict):
        self.root_dir = root_dir
        # self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config
        self.fake_feat = torch.from_numpy(fake_dict["fake_feat"]).float().unsqueeze(1)
        if config["dataset_args"]["unknown"] == True:
            self.fake_targets = [int(config["dataset_args"]["seen_num"]) for _ in range(len(fake_dict["fake_targets"]))]
            self.fake_text = [static_unknown_text for _ in range(len(fake_dict["fake_targets"]))]
        else:
            self.fake_targets = fake_dict["fake_targets"]
            self.fake_text = fake_dict["fake_text"]
        self.fake_vis = np.ones((len(self.fake_targets),))
        self.real_number = fake_dict["real_number"]
        self.meta_data_len = fake_dict["real_number"] + len(self.fake_targets)
    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.real_number:
            idx = idx - self.real_number
            y = self.fake_targets[idx]
            out_text = self.fake_text[idx]
            x = self.fake_feat[idx]
            vis_type = self.fake_vis[idx]
        else:
            y = self.data_list[idx]["label"]

            if self.config["dataset_args"]["original_text"] == True:
                out_text = self.data_list[idx]["text"]
            else:
                out_text = self.data_list[idx]["opt_text"]

            data_idx = self.data_list[idx]["file"]
            x = self.all_data[data_idx]

            vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type

class mmwave_new_Dataset(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data):
        self.root_dir = root_dir
        self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        # if config["dataset_args"]["dataset"] == "widar":
        #     original_first_dim_shape = all_data.shape[0]
        #     all_data = all_data.reshape(original_first_dim_shape, 22, 20, 20)
        #     self.all_data = torch.from_numpy(all_data).float()
        # else:
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config

    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.data_list[idx]["label"]

        if self.config["dataset_args"]["original_text"] == True:
            out_text = self.data_list[idx]["text"]
        else:
            out_text = self.data_list[idx]["opt_text"]

        data_idx = self.data_list[idx]["file"]
        x = self.all_data[data_idx]

        vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type


class mmwave_new_Dataset_fake(Dataset):
    def __init__(self, config, root_dir, meta_data, all_data, fake_dict):
        self.root_dir = root_dir

        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
        self.all_data = torch.from_numpy(all_data).float().unsqueeze(1)
        self.config = config
        self.fake_feat = torch.from_numpy(fake_dict["fake_feat"]).float().unsqueeze(1)
        if config["dataset_args"]["unknown"] == True:
            self.fake_targets = [int(config["dataset_args"]["seen_num"]) for _ in range(len(fake_dict["fake_targets"]))]
            self.fake_text = [static_unknown_text for _ in range(len(fake_dict["fake_targets"]))]
        else:
            self.fake_targets = fake_dict["fake_targets"]
            self.fake_text = fake_dict["fake_text"]
        self.fake_vis = np.ones((len(self.fake_targets),))
        self.real_number = fake_dict["real_number"]
        self.meta_data_len = fake_dict["real_number"] + len(self.fake_targets)


    def __len__(self):
        return self.meta_data_len

    def __getitem__(self, idx):
        """
        return data, label(idx), label(text)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= self.real_number:
            idx = idx - self.real_number
            y = self.fake_targets[idx]
            out_text = self.fake_text[idx]
            x = self.fake_feat[idx]
            vis_type = self.fake_vis[idx]
        else:
            y = self.data_list[idx]["label"]

            if self.config["dataset_args"]["original_text"] == True:
                out_text = self.data_list[idx]["text"]
            else:
                out_text = self.data_list[idx]["opt_text"]

            data_idx = self.data_list[idx]["file"]
            x = self.all_data[data_idx]

            vis_type = self.data_list[idx]["vis"]


        return x, out_text, y, vis_type
