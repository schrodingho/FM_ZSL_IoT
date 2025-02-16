import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.io as sio
import random
import os
import copy
import dill
import pandas as pd
from data_utils.gpt_aug import mmwave_GPT_AUG

static_unknown_text = "Unknown action"

def load_act_label():
    act_label_path = "./data_cache/MMFi_act.csv"
    act_df = pd.read_csv(act_label_path, sep="\t")
    # print(act_label)
    # Build the dictionary between activity and description
    act_code_desp_dict = act_df.set_index('Activity')['Description'].to_dict()
    return act_code_desp_dict

def list_src_act_folder(src_folder, modal_name, select_env=None, raw=False):
    # list all folders
    if modal_name == "mmwave":
        pass
    elif modal_name == "wifi":
        modal_name = "wifi-csi"
    assert modal_name in ["mmwave", "wifi-csi"]
    all_text_idx_list = [f"A{i}" if i >= 10 else f"A0{i}" for i in range(1, 28)]
    act_dict = {all_text_idx_list[i] : [] for i in range(len(all_text_idx_list))}
    if select_env == None:
        env_list = os.listdir(src_folder)
    else:
        env_list = select_env
    for env in env_list:
        env_path = os.path.join(src_folder, env)
        subj_list = os.listdir(env_path)
        for subj in subj_list:
            subj_path = os.path.join(env_path, subj)
            act_list = os.listdir(subj_path)
            for act in act_list:
                if raw == True:
                    act_path = os.path.join(subj_path, act, modal_name)
                else:
                    act_path = os.path.join(subj_path, act)
                act_dict[act].append(act_path)
    return act_dict

class general_meta_build:
    def __init__(self, config, dataset_name, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.label_text_dict = {}
        self.text_label_dict = {}
        self.unknown_class_text = static_unknown_text
        self.mapping_opt = {}
        self.dataset_name = dataset_name
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def gpt_aug(self, action_list):
        if self.config["dataset_args"]["dataset"] == "mmwave":
            for i in range(len(action_list)):
                action_list[i] = mmwave_GPT_AUG[action_list[i]]

        return action_list

    def bind_label_text(self, all_text_labels):
        for i in range(len(all_text_labels)):
            self.label_text_dict[i] = all_text_labels[i]

        self.text_label_dict = {v: k for k, v in self.label_text_dict.items()}

    def bind_label_text_dict(self, label_text_dict):
        self.label_text_dict = label_text_dict
        self.text_label_dict = {v: k for k, v in self.label_text_dict.items()}

    def optimize_text(self, text_list, prefix="Human action of ", suffix=" action"):
        if self.config["dataset_args"]["dataset"] == "mmwave" and self.config["dataset_args"]["gpt_aug"]:
            self.mapping_opt = mmwave_GPT_AUG
        else:
            for v in text_list:
                if not prefix or not suffix:
                    self.mapping_opt[v] = v
                else:
                    self.mapping_opt[v] = v + suffix
        print("[***] Text optimized: ", self.mapping_opt)


    def meta_building_no_unknown(self, seen_num, src_folder_path, select_env=None, raw=False):
        act_folders_path_dict = list_src_act_folder(src_folder_path, self.dataset_name, select_env=select_env, raw=raw)

        all_text_idx_list = list(self.text_label_dict.keys())

        all_text_idx_list = sorted(all_text_idx_list)
        random.shuffle(all_text_idx_list)

        all_text_idx_list = [f"A{i}" if i >= 10 else f"A0{i}" for i in range(1, 28)]
        act_code_text_dict = load_act_label()
        text_act_code_dict = {v: k for k, v in act_code_text_dict.items()}
        all_text_idx_list = [act_code_text_dict[i] for i in all_text_idx_list]

        self.optimize_text(all_text_idx_list, prefix=None, suffix=None)

        random.shuffle(all_text_idx_list)

        seen_text_pool = all_text_idx_list[:seen_num]
        unseen_text_pool = all_text_idx_list[seen_num:]

        if self.config["dataset_args"]["select_unseen"]:
            unseen_text_pool = self.config["dataset_args"]["select_unseen"]
            seen_text_pool = [text for text in all_text_idx_list if text not in unseen_text_pool]

        seen_text_label_dict = {seen_text_pool[i]: i for i in range(len(seen_text_pool))}
        # TODO: genTextList
        self.genTextList(seen_text_label_dict, type="train")

        # unseen_text_pool = all_text_idx_list[seen_num:]
        unseen_text_label_dict = {unseen_text_pool[i]: i for i in range(len(unseen_text_pool))}
        self.genTextList(unseen_text_label_dict, type="val")

        ###########################################################################################

        seen_act = []
        unseen_act = []
        unknown_act = []

        for text in seen_text_pool:
            seen_act.append(text_act_code_dict[text])
        for text in unseen_text_pool:
            unseen_act.append(text_act_code_dict[text])

        known_seen_meta = []
        unseen_meta = []
        seen_meta_class_dict = {}

        for act in seen_act:
            corres_text = act_code_text_dict[act]
            for act_folder in act_folders_path_dict[act]:
                # list all files absolute path in act_folder
                act_files = os.listdir(act_folder)
                for file in act_files:
                    idx = seen_text_label_dict[corres_text]
                    if idx not in seen_meta_class_dict:
                        seen_meta_class_dict[idx] = []
                    file_path = act_folder + "/" + file
                    known_seen_meta.append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                            "opt_text": self.mapping_opt[corres_text], "file": file_path, "vis": 0})

                    seen_meta_class_dict[seen_text_label_dict[corres_text]].append(
                        {"label": seen_text_label_dict[corres_text], "text": corres_text,
                         "opt_text": self.mapping_opt[corres_text], "file": file_path, "vis": 0})

        for act in unseen_act:
            corres_text = act_code_text_dict[act]
            for act_folder in act_folders_path_dict[act]:
                # list all files in act_folder
                act_files = os.listdir(act_folder)
                for file in act_files:
                    file_path = act_folder + "/" + file
                    unseen_meta.append({"label": unseen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": file_path, "vis": 1})



        ###########################################################################################
        train_meta = {"type": "train", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_tune_meta = {"type": "val_tune", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_seen_meta = {"type": "val_seen", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_unseen_meta = {"type": "val_unseen", "samples": 0, "label_text_dict": unseen_text_label_dict,
                           "data_list": []}

        assert self.train_ratio + self.val_ratio + self.test_ratio == 1

        for k, class_data in seen_meta_class_dict.items():
            random.shuffle(class_data)
            train_meta["data_list"].extend(class_data[:int(len(class_data) * self.train_ratio)])
            val_tune_meta["data_list"].extend(class_data[int(len(class_data) * self.train_ratio):int(
                len(class_data) * (self.train_ratio + self.val_ratio))])
            val_seen_meta["data_list"].extend(class_data[int(len(class_data) * (self.train_ratio + self.val_ratio)):])

        random.shuffle(train_meta["data_list"])
        train_meta["samples"] = len(train_meta["data_list"])
        val_seen_meta["samples"] = len(val_seen_meta["data_list"])

        val_tune_meta["samples"] = len(val_tune_meta["data_list"])

        val_unseen_meta["data_list"] = unseen_meta
        val_unseen_meta["samples"] = len(val_unseen_meta["data_list"])

        print("[***] New Train Test Split:")
        print(f"train_meta_samples: {train_meta['samples']}")

        print(f"val_tune_meta_samples: {val_tune_meta['samples']}")

        print(f"val_seen_meta_samples: {val_seen_meta['samples']}")
        print(f"val_unseen_meta_samples: {val_unseen_meta['samples']}")

        print(f"seen classes {seen_text_pool}")
        print(f"unseen classes {unseen_text_pool}")

        dill.dump(train_meta, open(f"{self.log_dir}/train_meta.pkl", "wb"))
        dill.dump(val_tune_meta, open(f"{self.log_dir}/val_tune_meta.pkl", "wb"))
        dill.dump(val_seen_meta, open(f"{self.log_dir}/val_seen_meta.pkl", "wb"))
        dill.dump(val_unseen_meta, open(f"{self.log_dir}/val_unseen_meta.pkl", "wb"))



    def meta_supervised(self, src_folder_path, select_env=None, raw=False):
        act_folders_path_dict = list_src_act_folder(src_folder_path, self.dataset_name, select_env=select_env, raw=raw)

        all_text_idx_list = list(self.text_label_dict.keys())

        all_text_idx_list = sorted(all_text_idx_list)
        random.shuffle(all_text_idx_list)

        all_text_idx_list = [f"A{i}" if i >= 10 else f"A0{i}" for i in range(1, 28)]
        act_code_text_dict = load_act_label()
        text_act_code_dict = {v: k for k, v in act_code_text_dict.items()}
        all_text_idx_list = [act_code_text_dict[i] for i in all_text_idx_list]

        self.optimize_text(all_text_idx_list, prefix=None, suffix=None)

        random.shuffle(all_text_idx_list)

        seen_text_pool = all_text_idx_list
        seen_text_label_dict = {seen_text_pool[i]: i for i in range(len(seen_text_pool))}


        ###########################################################################################

        seen_act = []

        for text in seen_text_pool:
            seen_act.append(text_act_code_dict[text])

        seen_meta_class_dict = {}

        for act in seen_act:
            corres_text = act_code_text_dict[act]
            for act_folder in act_folders_path_dict[act]:
                # list all files absolute path in act_folder
                act_files = os.listdir(act_folder)
                for file in act_files:
                    idx = seen_text_label_dict[corres_text]
                    if idx not in seen_meta_class_dict:
                        seen_meta_class_dict[idx] = []
                    file_path = act_folder + "/" + file

                    seen_meta_class_dict[seen_text_label_dict[corres_text]].append(
                        {"label": seen_text_label_dict[corres_text], "text": corres_text,
                         "opt_text": self.mapping_opt[corres_text], "file": file_path, "vis": 0})

        ###########################################################################################
        train_meta = {"type": "train", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_seen_meta = {"type": "val_seen", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}

        assert self.train_ratio + self.val_ratio + self.test_ratio == 1

        for k, class_data in seen_meta_class_dict.items():
            random.shuffle(class_data)
            train_meta["data_list"].extend(class_data[:int(len(class_data) * self.train_ratio)])
            val_seen_meta["data_list"].extend(class_data[int(len(class_data) * self.train_ratio):])

        random.shuffle(train_meta["data_list"])
        train_meta["samples"] = len(train_meta["data_list"])
        val_seen_meta["samples"] = len(val_seen_meta["data_list"])

        print("[***] New Train Test Split:")
        print(f"train_meta_samples: {train_meta['samples']}")

        print(f"val_seen_meta_samples: {val_seen_meta['samples']}")

        print(f"seen classes {seen_text_pool}")

        dill.dump(train_meta, open(f"{self.log_dir}/train_meta.pkl", "wb"))
        dill.dump(val_seen_meta, open(f"{self.log_dir}/val_seen_meta.pkl", "wb"))


    def genTextList(self, text_label_dict, type="train"):
        if not self.mapping_opt:
            raise ValueError("mapping_opt is None")
        # TODO: automatically mkdir
        # TODO: the generation can be incorrect
        textlist_father_dir = self.log_dir
        if not os.path.exists(textlist_father_dir):
            os.mkdir(textlist_father_dir)
        textlist_dir = textlist_father_dir + f"/{self.dataset_name}_{type}.txt"

        with open(textlist_dir, "w") as f:
            f.write("")
        with open(textlist_dir, "w") as f:
            for text in text_label_dict.keys():
                if text not in self.mapping_opt:
                    f.write(text + "\n")
                else:
                    f.write(self.mapping_opt[text] + "\n")


if __name__ == '__main__':
    # tmp1()
    pass
