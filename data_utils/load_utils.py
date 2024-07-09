import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import dill
import scipy.io as scio
from data_utils.base import static_unknown_text

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
        self.fake_feat = torch.from_numpy(fake_dict["fake_feat"]).float()
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


class WiFi_Dataset(Dataset):
    def __init__(self, config, root_dir, meta_data):
        self.root_dir = root_dir
        self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
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

        frame_path = self.data_list[idx]["file"]
        vis_type = self.data_list[idx]["vis"]

        data = scio.loadmat(frame_path)['CSIamp']
        data[np.isinf(scio.loadmat(frame_path)['CSIamp'])] = np.nan
        for i in range(10):  # 32
            temp_col = data[:, :, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

        # csi_amp = temp_col
        # df_csi_amp = pd.DataFrame(csi_amp)
        # pd.DataFrame.fillna(methode='ffill')
        # csi_amp = df_csi_amp.values

        data = torch.tensor((data - np.min(data)) / (np.max(data) - np.min(data)))
        data = np.array(data)

        return data, out_text, y, vis_type

class mmWave_Dataset(Dataset):
    def __init__(self, config, root_dir, meta_data):
        self.root_dir = root_dir
        self.meta_data_len = meta_data["samples"]
        # train or test
        self.type = meta_data["type"]
        self.data_list = meta_data["data_list"]
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

        frame_path = self.data_list[idx]["file"]

        with open(frame_path, 'rb') as f:
            raw_data = f.read()
            x = np.frombuffer(raw_data, dtype=np.float64)
            x = x.copy().reshape(-1, 5)

        vis_type = self.data_list[idx]["vis"]

        return x, out_text, y, vis_type

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

def setup_mmwave_data_loader(config, root_dir, train_meta, val_tune_meta, val_mix_meta, fake_dict=None):
    train_dataset = mmWave_Dataset(config, root_dir, train_meta)
    val_tune_dataset = mmWave_Dataset(config, root_dir, val_tune_meta)
    val_mix_dataset = mmWave_Dataset(config, root_dir, val_mix_meta)
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

def setup_wifi_data_loader(config, root_dir, train_meta, val_tune_meta, val_seen_meta, val_unseen_meta, val_mix_meta, fake_dict=None):
    train_dataset = WiFi_Dataset(config, root_dir, train_meta)
    val_tune_dataset = WiFi_Dataset(config, root_dir, val_tune_meta)
    val_seen_dataset = WiFi_Dataset(config, root_dir, val_seen_meta)
    val_unseen_dataset = WiFi_Dataset(config, root_dir, val_unseen_meta)
    val_mix_dataset = WiFi_Dataset(config, root_dir, val_mix_meta)
    return [train_dataset, val_tune_dataset, val_seen_dataset, val_unseen_dataset, val_mix_dataset]


Dataset_Dict = {
    "USC": USC_Dataset,
    "mmwave": mmWave_Dataset,
    "wifi": WiFi_Dataset
}

def setup_supervised_loader(config, root_dir, train_meta, val_meta, all_data=None):
    if config["dataset_args"]["dataset"] == "USC":
        train_dataset = USC_Dataset(config, root_dir, train_meta, all_data)
        val_dataset = USC_Dataset(config, root_dir, val_meta, all_data)
    else:
        train_dataset = Dataset_Dict[config["dataset_args"]["dataset"]](root_dir, train_meta)
        val_dataset = Dataset_Dict[config["dataset_args"]["dataset"]](root_dir, val_meta)
    return train_dataset, val_dataset

def collate_fn_padd(batch):

    ## get sequence lengths
    ## padd


    x = [torch.Tensor(t[0]) for t in batch]
    x = torch.nn.utils.rnn.pad_sequence(x)
    x = x.permute(1,0,2)
    out_text = [t[1] for t in batch]
    y = torch.LongTensor([t[2] for t in batch])
    vis_type = torch.LongTensor([t[3] for t in batch])

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
