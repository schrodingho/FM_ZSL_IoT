import torch
import numpy as np
import torch.optim as optim
from utils.other import FastDataLoader
from argparse import ArgumentParser
from utils.sup_model import SupModel

from data_utils.base import merge_txt_files
from data_utils.load_utils import (setup_imu_data_loader,
                                   setup_pamap_data_loader,
                                setup_mmwave_new_data_loader)
from data_utils.dataset_reschedule import mix_val_data
from extension.local_train import train_supervised_model

import dill
import logging
import time
import os
import ruamel.yaml as yaml

import data_utils.datasets_entry
import pandas as pd
from data_utils.extract_raw_data import extract_raw_func


def main(args):
    torch.cuda.empty_cache()
    config_path = f"./settings/{args.config_choose}.yaml"

    with open(config_path, "r") as f:
        yaml_loader = yaml.YAML(typ="safe")
        config = yaml_loader.load(f)

    if args.back_up_path is not None:
        config["dataset_args"]["backup"] = args.back_up_path

    use_back_up = config["dataset_args"]["backup"]
    dataset_name = config["dataset_args"]["dataset"]
    config["args"]["save"] = True
    new_log_dir = f"./logs/logs_{dataset_name}/"
    current_time = time.strftime("%Y%m%d-%H%M%S")

    current_log = new_log_dir + "logdir_" + current_time
    if not os.path.exists(current_log):
        os.makedirs(current_log)
    config["model_path"] = current_log

    # meta_data generation
    if use_back_up is None:
        data_utils.datasets_entry.meta_generate(config)

    np.random.seed(config["args"]["seed"])
    torch.manual_seed(config["args"]["seed"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(filename=current_log + f"/experiment.log",
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    open_set_df = pd.DataFrame(columns=["Epoch", "OSR", "FPR", "FNR", "TNR", "TPR", "precision", "recall", "fscore"])
    metrics_df = pd.DataFrame(columns=["Epoch", "AvgRecall", "AvgPrecision", "AvgF1", "Unseen Acc"])
    no_open_metrics_df = pd.DataFrame(columns=["Epoch", "AvgRecall", "AvgPrecision", "AvgF1", "Unseen Acc"])
    gzsl_metrics_df = pd.DataFrame(columns=["S", "U", "H"])
    gzsl_no_open_metrics_df = pd.DataFrame(columns=["S", "U", "H"])

    config["args"]["open_set_df"] = open_set_df
    config["args"]["metrics_df"] = metrics_df
    config["args"]["no_open_metrics_df"] = no_open_metrics_df
    config["args"]["gzsl_metrics_df"] = gzsl_metrics_df
    config["args"]["gzsl_no_open_metrics_df"] = gzsl_no_open_metrics_df

    root_dir = None

    if use_back_up is None:
        #*************** generate new dataset for training  ***************#
        train_meta = dill.load(open(f"{current_log}/train_meta.pkl", "rb"))
        val_tune_meta = dill.load(open(f"{current_log}/val_tune_meta.pkl", "rb"))
        val_seen_meta = dill.load(open(f"{current_log}/val_seen_meta.pkl", "rb"))
        val_unseen_meta = dill.load(open(f"{current_log}/val_unseen_meta.pkl", "rb"))

        all_data = np.load(open(f"data_cache/{dataset_name}_data.npy", "rb"))
        data_shape = all_data.shape
        config["dataset_args"]["train_shape"] = data_shape

        seen_meta_list = f"{current_log}/{dataset_name}_train.txt"
        unseen_meta_list = f"{current_log}/{dataset_name}_val.txt"
        merge_txt_files(seen_meta_list, unseen_meta_list, f"{current_log}/all_text.txt")
        all_meta_list = f"{current_log}/all_text.txt"

        val_mix_meta = mix_val_data(val_seen_meta, val_unseen_meta, method=4)
        dill.dump(val_mix_meta, open(f"{current_log}/mix_val_meta.pkl", "wb"))

        extract_dir = f"{current_log}/extracted_generator"
        config["extract_dir"] = extract_dir

        config["extract_raw_dir"] = f"{current_log}/extracted_raw"
        if not os.path.exists(config["extract_raw_dir"]):
            os.makedirs(config["extract_raw_dir"])

        extract_raw_func(config, train_meta, val_seen_meta, val_unseen_meta, seen_meta_list, unseen_meta_list, all_data)

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        if config["dataset_args"]["fake"] == True:
            extract_raw = f"{current_log}/extracted_raw"
            config["extract_raw_dir"] = extract_raw
            if not os.path.exists(extract_raw):
                os.makedirs(extract_raw)
            # extract raw
            gen_fake_text = dill.load(open(f"{extract_raw}/unseen_fake_text.pkl", "rb"))
            gen_fake_feat = np.load(f"{extract_raw}/unseen_fake_raw_data.npy")
            gen_fake_targets = np.load(f"{extract_raw}/unseen_fake_targets.npy")
            fake_dict = {"fake_text": gen_fake_text, "fake_feat": gen_fake_feat, "fake_targets": gen_fake_targets}
    else:
        #*************** use the existing dataset for training or testing ***************#
        train_meta = dill.load(open(f"{use_back_up}/train_meta.pkl", "rb"))
        val_tune_meta = dill.load(open(f"{use_back_up}/val_tune_meta.pkl", "rb"))
        val_seen_meta = dill.load(open(f"{use_back_up}/val_seen_meta.pkl", "rb"))
        val_unseen_meta = dill.load(open(f"{use_back_up}/val_unseen_meta.pkl", "rb"))
        val_mix_meta = dill.load(open(f"{use_back_up}/mix_val_meta.pkl", "rb"))

        all_data = np.load(open(f"data_cache/{dataset_name}_data.npy", "rb"))
        data_shape = all_data.shape
        config["dataset_args"]["train_shape"] = data_shape

        seen_meta_list = f"{use_back_up}/{dataset_name}_train.txt"
        unseen_meta_list = f"{use_back_up}/{dataset_name}_val.txt"

        merge_txt_files(seen_meta_list, unseen_meta_list, f"{current_log}/all_text.txt")
        all_meta_list = f"{current_log}/all_text.txt"

        extract_dir = f"{current_log}/extracted_generator"
        config["extract_dir"] = extract_dir

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        config["extract_raw_dir"] = f"{current_log}/extracted_raw"
        if not os.path.exists(config["extract_raw_dir"]):
            os.makedirs(config["extract_raw_dir"])

        #*************** data augmentation entry ***************#
        if config["dataset_args"]["fake"] == True:
            extract_raw = f"{current_log}/extracted_raw"
            config["extract_raw_dir"] = extract_raw
            if not os.path.exists(extract_raw):
                os.makedirs(extract_raw)
            # extract raw
            gen_fake_text = dill.load(open(f"{use_back_up}/extracted_raw/unseen_fake_text.pkl", "rb"))
            gen_fake_feat = np.load(f"{use_back_up}/extracted_raw/unseen_fake_raw_data.npy")
            gen_fake_targets = np.load(f"{use_back_up}/extracted_raw/unseen_fake_targets.npy")
            fake_dict = {"fake_text": gen_fake_text, "fake_feat": gen_fake_feat, "fake_targets": gen_fake_targets}


    logging.info("train_samples: {}".format(train_meta["samples"]))
    logging.info("val_tune_samples: {}".format(val_tune_meta["samples"]))
    logging.info("val_seen_samples: {}".format(val_seen_meta["samples"]))
    logging.info("val_unseen_samples: {}".format(val_unseen_meta["samples"]))
    logging.info("mix_meta_samples: {}".format(val_mix_meta["samples"]))

    logging.info("seen_set: {}".format(list(train_meta["label_text_dict"].keys())))
    logging.info("val_seen_set: {}".format(list(val_unseen_meta["label_text_dict"].keys())))

    # *************** initialize dataloaders ***************#
    if dataset_name == "USC":
        if config["dataset_args"]["fake"] == True:
            [trn_dataset, val_tune_dataset, val_mix_dataset,
             fake_dataset] = setup_imu_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data,
                fake_dict)
        else:
            [trn_dataset, val_tune_dataset,
             val_mix_dataset] = setup_imu_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data)
    elif dataset_name == "pamap":
        if config["dataset_args"]["fake"] == True:
            [trn_dataset, val_tune_dataset, val_mix_dataset,
             fake_dataset] = setup_pamap_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data,
                fake_dict)
        else:
            [trn_dataset, val_tune_dataset,
             val_mix_dataset] = setup_pamap_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data)
    elif dataset_name == "mmwave" or dataset_name == "wifi":
        if config["dataset_args"]["fake"] == True:
            [trn_dataset, val_tune_dataset,  val_mix_dataset,
             fake_dataset] = setup_mmwave_new_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data,
                fake_dict)
        else:
            [trn_dataset, val_tune_dataset,
             val_mix_dataset] = setup_mmwave_new_data_loader(
                config, root_dir, train_meta, val_tune_meta, val_mix_meta, all_data)
    else:
        raise ValueError("Dataset not found")

    batch_size = config["args"]["batchsize"]
    workers = config["args"]["workers"]

    trnloader = FastDataLoader(trn_dataset, batch_size=batch_size, num_workers=workers,
                               shuffle=True, pin_memory=False, drop_last=True)
    if config["args"]["test"]:
        trnloader = FastDataLoader(trn_dataset, batch_size=batch_size, num_workers=workers,
                                   shuffle=False, pin_memory=False, drop_last=False)

    val_tune_loader = FastDataLoader(val_tune_dataset, batch_size=batch_size, num_workers=workers,
                                     shuffle=False, pin_memory=False, drop_last=False)

    val_mix_loader = FastDataLoader(val_mix_dataset, batch_size=batch_size, num_workers=workers,
                                    shuffle=False, pin_memory=False, drop_last=False)

    # initialize models
    print('==> reading meta data for {}'.format(config["dataset_args"]["dataset"]))

    # *************** initialize the supervised model ***************#
    model = SupModel(config, device)
    model.float()
    model.to(device)

    # *************** initialize optimizer and scheduler ***************#
    optimizer = optim.SGD(model.parameters(), lr=config["args"]["lr"], momentum=0.9, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["args"]["decay_steps"])],
                                                        gamma=0.1)

    config["args"]["start_iter"] = 0

    print('======> start supervised training {}, use {}.'.format(dataset_name, device))
    train_supervised_model(config, [trnloader, val_tune_loader, val_mix_loader], model, optimizer, lr_scheduler, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_choose', type=str, default='pamap',
                        choices=['USC', 'wifi', 'mmwave', 'pamap'])
    parser.add_argument('--back_up_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
