import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.io as sio
import random
import os
import copy
import dill
from data_utils.gpt_aug import USC_GPT_AUG, pamap_GPT_AUG

static_unknown_text = "Unknown action"

MEAN_OF_IMU = [-0.32627436907665514, -0.8661114601303396]
STD_OF_IMU = [0.6761486428324216, 113.55369543559192]
MEAN_OF_SKELETON = [-0.08385579666058844, -0.2913725901521685, 2.8711066708996738]
STD_OF_SKELETON = [0.14206656362043646, 0.4722835954035046, 0.16206781976658088]

def UTD_MHAD_dict():
    all_actions = ["Swipe left", "Swipe right", "Wave", "Clap", "Throw", "Arm cross", "Basketball shoot", "Draw X",
                   "Draw circle (clockwise)", "Draw circle (counter clockwise)", "Draw triangle", "Bowling", "Boxing",
                   "Baseball Swing",
                   "Tennis Swing", "Arm Curl", "Tennis Serve", "Push", "Knock", "Catch", "Pickup Throw", "Jog",
                   "Walk", "Sit to stand", "Stand to sit", "Lunge", "Squat"]
    action_idx_dict = {i: all_actions[i] for i in range(len(all_actions))}
    return action_idx_dict

def PAMAP_dict():
    all_actions = ["Lying", "Sitting", "Standing", "Walking", "Running", "Cycling", "Nordic Walking",
                   "Ascending Stairs", "Descending Stairs", "Vacuum cleaning", "Ironing", "Rope Jumping"]
    action_idx_dict = {i: all_actions[i] for i in range(len(all_actions))}
    return action_idx_dict

def USC_HAD_dict():
    # GPT_AUG = [
    #     "Moving forward at a regular pace on foot.",
    #     "Moving to the left at a regular pace on foot.",
    #     "Moving to the right at a regular pace on foot.",
    #     "Moving upward on stairs at a regular pace.",
    #     "Moving downward on stairs at a regular pace.",
    #     "Moving rapidly forward on foot.",
    #     "Propelling oneself upward off the ground.",
    #     "Being seated with weight on buttocks and thighs.",
    #     "Upright position on feet without support.",
    #     "Resting in a horizontal position for rest.",
    #     "Moving upward in an elevator.",
    #     "Moving downward in an elevator."
    # ]
    all_actions = ["Walking Forward", "Walking Left", "Walking Right", "Walking Upstairs", "Walking Downstairs",
     "Running Forward", "Jumping Up", "Sitting", "Standing", "Sleeping", "Elevator Up", "Elevator Down"]
    action_idx_dict = {i: all_actions[i] for i in range(len(all_actions))}
    return action_idx_dict

class general_meta_build:
    def __init__(self, config, all_data, all_y_true, dataset_name, log_dir):
        self.config = config
        self.all_data = all_data
        self.all_y_true = all_y_true
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
        if self.config["dataset_args"]["dataset"] == "USC":
            for i in range(len(action_list)):
                action_list[i] = USC_GPT_AUG[action_list[i]]

        if self.config["dataset_args"]["dataset"] == "pamap":
            for i in range(len(action_list)):
                action_list[i] = pamap_GPT_AUG[action_list[i]]

        return action_list


    def bind_label_text(self, all_text_labels):
        for i in range(len(all_text_labels)):
            self.label_text_dict[i] = all_text_labels[i]

        self.text_label_dict = {v: k for k, v in self.label_text_dict.items()}

    def bind_label_text_dict(self, label_text_dict):
        self.label_text_dict = label_text_dict
        self.text_label_dict = {v: k for k, v in self.label_text_dict.items()}
    def optimize_text(self, prefix="Human action of ", suffix=" action"):
        # if self.config["dataset_args"]["dataset"] == "USC" and self.config["dataset_args"]["gpt_aug"]:
        #     self.mapping_opt = USC_GPT_AUG
        # elif self.config["dataset_args"]["dataset"] == "pamap" and self.config["dataset_args"]["gpt_aug"]:
        #     self.mapping_opt = pamap_GPT_AUG
        # else:
        if not self.label_text_dict:
            raise ValueError("label_text_dict is None")
        for k, v in self.label_text_dict.items():
            if not prefix and not suffix:
                self.mapping_opt[v] = v
            else:
                if not prefix and suffix:
                    self.mapping_opt[v] = v + suffix
                if not suffix and prefix:
                    self.mapping_opt[v] = prefix + v
                if prefix and suffix:
                    self.mapping_opt[v] = prefix + v + suffix
        print("[***] Text optimized: ", self.mapping_opt)

    def meta_building(self, seen_num, known_num):
        distinct_labels = np.unique(self.all_y_true)
        label_num = len(distinct_labels)
        if not self.label_text_dict or not self.text_label_dict:
            raise ValueError("label_text_dict is None")

        if not self.mapping_opt:
            raise ValueError("mapping_opt is None")

        all_text_idx_list = list(self.text_label_dict.keys())
        assert label_num == len(all_text_idx_list)

        all_text_idx_list = sorted(all_text_idx_list)
        random.shuffle(all_text_idx_list)

        seen_text_pool = all_text_idx_list[:seen_num]

        # seen_text_pool = ['Jumping Up', 'Walking Downstairs', 'Sitting', 'Walking Right', 'Elevator Up', 'Walking Left',
        #          'Running Forward', 'Sleeping', 'Elevator Down']

        known_seen_text_pool = seen_text_pool[:known_num]

        seen_text_label_dict = {known_seen_text_pool[i]: i for i in range(len(known_seen_text_pool))}
        largest_label_seen = int(max(seen_text_label_dict.values()))
        unknown_class_label = largest_label_seen + 1
        real_seen_text_label_dict = copy.deepcopy(seen_text_label_dict)
        real_seen_text_label_dict[self.unknown_class_text] = unknown_class_label

        self.genTextList(real_seen_text_label_dict, type="train")


        unknown_text_pool = []
        for text in seen_text_pool:
            if text not in known_seen_text_pool:
                unknown_text_pool.append(text)


        unseen_text_pool = all_text_idx_list[seen_num:]
        # unseen_text_pool = ['Walking Forward', 'Standing', 'Walking Upstairs']
        # ['Walking Forward', 'Standing', 'Walking Upstairs']

        unseen_text_label_dict = {unseen_text_pool[i]: i for i in range(len(unseen_text_pool))}
        self.genTextList(unseen_text_label_dict, type="val")

        ###########################################################################################

        # if not self.all_data or not self.all_y_true:
        #     raise ValueError("all_data or all_labels is None")

        seen_label_idx = []
        unseen_label_idx = []
        unknown_idx = []

        for text in seen_text_pool:
            if text in unknown_text_pool:
                unknown_idx.append(self.text_label_dict[text])
            else:
                seen_label_idx.append(self.text_label_dict[text])
        for text in unseen_text_pool:
            unseen_label_idx.append(self.text_label_dict[text])

        known_seen_meta = []
        unknown_seen_meta = []
        unseen_meta = []
        seen_meta_class_dict = {}

        for idx in seen_label_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            if idx not in seen_meta_class_dict:
                seen_meta_class_dict[idx] = []
            for target_idx in target_idx_list:
                known_seen_meta.append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 0})
                seen_meta_class_dict[idx].append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 0})

        for idx in unseen_label_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            for target_idx in target_idx_list:
                unseen_meta.append({"label": unseen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 1})

        for idx in unknown_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            for target_idx in target_idx_list:
                unknown_seen_meta.append({"label": unknown_class_label, "text": corres_text,
                                        "opt_text": self.unknown_class_text, "file": target_idx, "vis": 1})

        ###########################################################################################

        train_meta = {"type": "train", "samples": 0, "label_text_dict": real_seen_text_label_dict, "data_list": []}
        val_tune_meta = {"type": "val_tune", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_seen_meta = {"type": "val_seen", "samples": 0, "label_text_dict": seen_text_label_dict, "data_list": []}
        val_unseen_meta = {"type": "val_unseen", "samples": 0, "label_text_dict": unseen_text_label_dict,
                           "data_list": []}


        assert self.train_ratio + self.val_ratio + self.test_ratio == 1

        for k, class_data in seen_meta_class_dict.items():
            random.shuffle(class_data)
            train_meta["data_list"].extend(class_data[:int(len(class_data) * self.train_ratio)])
            val_tune_meta["data_list"].extend(class_data[int(len(class_data) * self.train_ratio):int(len(class_data) * (self.train_ratio + self.val_ratio))])
            val_seen_meta["data_list"].extend(class_data[int(len(class_data) * (self.train_ratio + self.val_ratio)):])


        train_meta["data_list"].extend(unknown_seen_meta)
        random.shuffle(train_meta["data_list"])
        train_meta["samples"] = len(train_meta["data_list"])
        val_seen_meta["samples"] = len(val_seen_meta["data_list"])

        val_tune_meta["samples"] = len(val_tune_meta["data_list"])

        val_unseen_meta["data_list"] = unseen_meta
        val_unseen_meta["samples"] = len(val_unseen_meta["data_list"])

        print("[***] New Train Test Split:")
        print(f"train_meta_samples: {train_meta['samples']}")
        print(f"unknown samples in train_meta: {len(unknown_seen_meta)}")

        print(f"val_tune_meta_samples: {val_tune_meta['samples']}")

        print(f"val_seen_meta_samples: {val_seen_meta['samples']}")
        print(f"val_unseen_meta_samples: {val_unseen_meta['samples']}")

        print(f"seen classes {seen_text_pool}")
        print(f"unknown classes in seen classes {unknown_text_pool}")
        print(f"unseen classes {unseen_text_pool}")

        dill.dump(train_meta, open(f"{self.log_dir}/train_meta.pkl", "wb"))
        dill.dump(val_tune_meta, open(f"{self.log_dir}/val_tune_meta.pkl", "wb"))

        dill.dump(val_seen_meta, open(f"{self.log_dir}/val_seen_meta.pkl", "wb"))
        dill.dump(val_unseen_meta, open(f"{self.log_dir}/val_unseen_meta.pkl", "wb"))


    def meta_building_no_unknown(self, seen_num):
        distinct_labels = np.unique(self.all_y_true)
        label_num = len(distinct_labels)
        if not self.label_text_dict or not self.text_label_dict:
            raise ValueError("label_text_dict is None")

        if not self.mapping_opt:
            raise ValueError("mapping_opt is None")

        all_text_idx_list = list(self.text_label_dict.keys())
        assert label_num == len(all_text_idx_list)

        all_text_idx_list = sorted(all_text_idx_list)
        random.shuffle(all_text_idx_list)

        seen_text_pool = all_text_idx_list[:seen_num]
        unseen_text_pool = all_text_idx_list[seen_num:]

        if self.config["dataset_args"]["select_unseen"]:
            unseen_text_pool = self.config["dataset_args"]["select_unseen"]
            seen_text_pool = [text for text in all_text_idx_list if text not in unseen_text_pool]

        # seen_text_pool = ['Jumping Up', 'Walking Downstairs', 'Sitting', 'Walking Right', 'Elevator Up', 'Walking Left',
        #          'Running Forward', 'Sleeping', 'Elevator Down']

        seen_text_label_dict = {seen_text_pool[i]: i for i in range(len(seen_text_pool))}
        # TODO: genTextList
        self.genTextList(seen_text_label_dict, type="train")

        # unseen_text_pool = ['Walking Forward', 'Standing', 'Walking Upstairs']


        unseen_text_label_dict = {unseen_text_pool[i]: i for i in range(len(unseen_text_pool))}
        self.genTextList(unseen_text_label_dict, type="val")

        ###########################################################################################

        # if not self.all_data or not self.all_y_true:
        #     raise ValueError("all_data or all_labels is None")

        seen_label_idx = []
        unseen_label_idx = []

        for text in seen_text_pool:
            seen_label_idx.append(self.text_label_dict[text])
        for text in unseen_text_pool:
            unseen_label_idx.append(self.text_label_dict[text])

        known_seen_meta = []
        unseen_meta = []
        seen_meta_class_dict = {}

        for idx in seen_label_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            if idx not in seen_meta_class_dict:
                seen_meta_class_dict[idx] = []
            for target_idx in target_idx_list:
                known_seen_meta.append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 0})
                seen_meta_class_dict[idx].append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 0})

        for idx in unseen_label_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            for target_idx in target_idx_list:
                unseen_meta.append({"label": unseen_text_label_dict[corres_text], "text": corres_text,
                                        "opt_text": self.mapping_opt[corres_text], "file": target_idx, "vis": 1})

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
            val_tune_meta["data_list"].extend(
                class_data[
                int(len(class_data) * self.train_ratio):int(len(class_data) * (self.train_ratio + self.val_ratio))])
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

    def meta_supervised(self):
        distinct_labels = np.unique(self.all_y_true)
        label_num = len(distinct_labels)
        if not self.label_text_dict or not self.text_label_dict:
            raise ValueError("label_text_dict is None")

        if not self.mapping_opt:
            raise ValueError("mapping_opt is None")

        all_text_idx_list = list(self.text_label_dict.keys())
        assert label_num == len(all_text_idx_list)

        all_text_idx_list = sorted(all_text_idx_list)
        random.shuffle(all_text_idx_list)

        seen_text_pool = all_text_idx_list
        seen_text_label_dict = {seen_text_pool[i]: i for i in range(len(seen_text_pool))}

        seen_label_idx = []

        for text in seen_text_pool:
            seen_label_idx.append(self.text_label_dict[text])

        seen_meta_class_dict = {}

        for idx in seen_label_idx:
            corres_text = self.label_text_dict[idx]
            target_idx_list = np.where(self.all_y_true == idx)[0]
            if idx not in seen_meta_class_dict:
                seen_meta_class_dict[idx] = []
            for target_idx in target_idx_list:
                seen_meta_class_dict[idx].append({"label": seen_text_label_dict[corres_text], "text": corres_text,
                                                  "opt_text": self.mapping_opt[corres_text], "file": target_idx,
                                                  "vis": 0})

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
        textlist_other_dir = textlist_father_dir + f"/{self.dataset_name}_other_{type}.txt"
        with open(textlist_dir, "w") as f:
            f.write("")
        with open(textlist_dir, "w") as f:
            for text in text_label_dict.keys():
                if text not in self.mapping_opt:
                    f.write(text + "\n")
                else:
                    f.write(self.mapping_opt[text] + "\n")

        with open(textlist_other_dir, "w") as f:
            f.write("")
        with open(textlist_other_dir, "w") as f:
            for text in text_label_dict.keys():
                    f.write(text + "\n")

if __name__ == '__main__':
    # tmp1()
    utd_dataset_path = "/home/dingding/Datasets/Inertial"
