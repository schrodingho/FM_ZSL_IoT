import time
import os
import torch
import random
import numpy as np
import pandas as pd
from utils import save_checkpoint, save_best_checkpoint, save_best_checkpoint_epoch, delete_redundant_epoch_dirs, save_unseen_best
from suploss import SupConLoss
import logging
from open_set import val_parameter_define, val_parameter_define_text
from validations import mix_validation, mix_validation_text
from tqdm import tqdm
from imu.datasets import static_unknown_text
from data_utils.gpt_aug import GPT_AUG_DICT

import warnings
warnings.filterwarnings("ignore")
from models.all_sup_model import SupModel


def test_CLIPrompt(config, dataloader, text, model, device):
    logging.info(
        f"KNN_VAL: {config['open_set_args']['knn_val']}, "
        f"KNN_THRESHOLD: {config['open_set_args']['knn_threshold']}"
    )

    trnloader, val_tune_loader, val_mix_loader = dataloader

    epoch = 0
    epochs = config["args"]["epochs"]
    open_set_df = config["args"]["open_set_df"]
    metrics_df = config["args"]["metrics_df"]
    no_open_metrics_df = config["args"]["no_open_metrics_df"]
    gzsl_metrics_df = config["args"]["gzsl_metrics_df"]
    gzsl_no_open_metrics_df = config["args"]["gzsl_no_open_metrics_df"]


    checkpoint_path = config["args"]["test_model_path"]
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    specialist_model = None
    if config["args"]["spe_path"] is not None:
        print("Specialist model is loaded")
        specialist_model = SupModel(config, device)
        specialist_model_path = config["args"]["spe_path"]
        specialist_model.load_state_dict(torch.load(specialist_model_path)['state_dict'])
        specialist_model.to(device)

    # TODO: 1. seen validation -> acc -> best_acc -> saving
    # val_model_seen(config, epoch, val_tune_loader, text, model, device)
    # TODO: 2. parameter tuning (training data sampling?)
    # if epoch >= 1 and (epoch + 1) % 4 == 0:

    model.eval()
    model.clipmodel.eval()


    known_trn_vFeatures, known_trn_targets = extract_trn_feat(config, epoch, trnloader, text, model, device)
    if config["open_set_args"]["manual"] == False:
        auto_knn_params = val_parameter_define(config, epoch, val_tune_loader, text, model, known_trn_vFeatures, known_trn_targets, device)
    else:
        auto_knn_params = None

    eval_dict, val_feat_pack = mix_validation(config, epoch, val_mix_loader, text, model, known_trn_vFeatures, known_trn_targets, device, auto_knn_params, specialist_model)
    [val_vFeatures, val_targets, val_vis_lists, pos_tFeature, neg_tFeature, gzsl_eval_dict] = val_feat_pack

    # select unseen
    zsl_unseen_1 = eval_dict["zsl_unseen_1"]
    zsl_unseen_2 = eval_dict["zsl_unseen_2"]
    gzsl_unseen = gzsl_eval_dict["zsl_unseen_2"]

    open_set = eval_dict["open_set"]
    open_set_dict = eval_dict["open_set_dict"]
    os_FPR = open_set_dict["FPR"]
    os_TPR = open_set_dict["TPR"]
    os_TNR = open_set_dict["TNR"]
    os_FNR = open_set_dict["FNR"]
    os_precision = open_set_dict["precision"]
    os_recall = open_set_dict["recall"]
    os_fscore = open_set_dict["fscore"]

    gzsl_s = eval_dict["gzsl_dict"]["S"].detach().cpu().numpy()
    gzsl_u = eval_dict["gzsl_dict"]["U"].detach().cpu().numpy()
    gzsl_h = eval_dict["gzsl_dict"]["H"].detach().cpu().numpy()

    gzsl_no_open_s = gzsl_eval_dict["gzsl_dict"]["S"].detach().cpu().numpy()
    gzsl_no_open_u = gzsl_eval_dict["gzsl_dict"]["U"].detach().cpu().numpy()
    gzsl_no_open_h = gzsl_eval_dict["gzsl_dict"]["H"].detach().cpu().numpy()

    print(f"Epoch: {epoch}, avg_recall:{eval_dict['avg_recall']}, avg_prec:{eval_dict['avg_precision']}, avg_f1:{eval_dict['avg_f1']}")
    logging.info(f"Epoch: {epoch}, avg_recall:{eval_dict['avg_recall']}, avg_prec:{eval_dict['avg_precision']}, avg_f1:{eval_dict['avg_f1']}")

    open_set_df.loc[len(open_set_df)] = [epoch, open_set.detach().cpu().numpy(), os_FPR, os_FNR, os_TNR, os_TPR, os_precision, os_recall, os_fscore]
    metrics_df.loc[len(metrics_df)] = [epoch, eval_dict['avg_recall'], eval_dict['avg_precision'], eval_dict['avg_f1'], zsl_unseen_2.detach().cpu().numpy()]
    no_open_metrics_df.loc[len(no_open_metrics_df)] = [epoch, gzsl_eval_dict['avg_recall'], gzsl_eval_dict['avg_precision'], gzsl_eval_dict['avg_f1'],
                                                       gzsl_unseen.detach().cpu().numpy()]
    gzsl_metrics_df.loc[len(gzsl_metrics_df)] = [gzsl_s, gzsl_u, gzsl_h]
    gzsl_no_open_metrics_df.loc[len(gzsl_no_open_metrics_df)] = [gzsl_no_open_s, gzsl_no_open_u, gzsl_no_open_h]

    open_set_df.to_csv(config["model_path"] + f"/open_set_{config['baseline_args']['baseline']}_test.csv", index=False)
    metrics_df.to_csv(config["model_path"] + f"/metrics_{config['baseline_args']['baseline']}_test.csv", index=False)
    no_open_metrics_df.to_csv(config["model_path"] + f"/no_open_metrics_{config['baseline_args']['baseline']}_test.csv", index=False)

    gzsl_metrics_df.to_csv(config["model_path"] + f"/gzsl_metrics_{config['baseline_args']['baseline']}_test.csv", index=False)
    gzsl_no_open_metrics_df.to_csv(config["model_path"] + f"/gzsl_no_open_metrics_{config['baseline_args']['baseline']}_test.csv", index=False)

    return

    # return best_open_acc

def extract_nol_feat(config, dataloader, text, model, device):
    print("#######EXTRACTED NOL FEAT######")
    trnloader, val_tune_loader, val_mix_loader = dataloader

    epoch = 0
    epochs = config["args"]["epochs"]
    open_set_df = config["args"]["open_set_df"]
    metrics_df = config["args"]["metrics_df"]
    no_open_metrics_df = config["args"]["no_open_metrics_df"]
    gzsl_metrics_df = config["args"]["gzsl_metrics_df"]
    gzsl_no_open_metrics_df = config["args"]["gzsl_no_open_metrics_df"]


    checkpoint_path = config["args"]["test_model_path"]
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    model.eval()
    model.clipmodel.eval()


    known_trn_vFeatures, known_trn_targets = extract_trn_feat(config, epoch, trnloader, text, model, device)
    if config["open_set_args"]["manual"] == False:
        auto_knn_params = val_parameter_define(config, epoch, val_tune_loader, text, model, known_trn_vFeatures, known_trn_targets, device)
    else:
        auto_knn_params = None

    eval_dict, val_feat_pack = mix_validation(config, epoch, val_mix_loader, text, model, known_trn_vFeatures, known_trn_targets, device, auto_knn_params)
    [val_vFeatures, val_targets, val_vis_lists, pos_tFeature, neg_tFeature, gzsl_eval_dict] = val_feat_pack

    zsl_feat_extraction(config, known_trn_vFeatures, known_trn_targets, val_vFeatures, val_targets, val_vis_lists, pos_tFeature, neg_tFeature)

def val_model_seen(config, epoch, dataloader, text, model, device):
    dataset_name = config["dataset_args"]["dataset"]
    model.eval()
    with torch.no_grad():
        actionlist, actiondict, actiontoken = text
        index_unknown = np.argwhere(actionlist == static_unknown_text)
        actionlist = np.delete(actionlist, index_unknown)

        gpt_aug_actionlist = [GPT_AUG_DICT[dataset_name][word] for word in actionlist]

        best_seen_acc = 0

        model.eval()
        with torch.no_grad():
            targets = torch.zeros(0).to(device)
            vFeature_lists = torch.zeros(0).to(device)
            for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
                vids, name, y_true, vis_type = sample
                if dataset_name == "USC":
                    vids = vids.unsqueeze(1)

                if idx == 0:
                    vFeature, tFeature, _ = model(vids.to(device), actionlist, gpt_aug_actionlist, type="all")
                    tFeature = tFeature[:config["dataset_args"]["seen_num"], :]
                    tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)
                else:
                    vFeature, _, _ = model(vids.to(device), actionlist[:1], gpt_aug_actionlist[:1], type="all")

                target_batch = y_true.to(device)

                vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)

                targets = torch.cat([targets, target_batch], dim=0)
                vFeature_lists = torch.cat([vFeature_lists, vFeature], dim=0)

            all_logits = vFeature_lists @ tFeature.t() / 0.07
            similarity = all_logits.softmax(dim=-1)
            values, indices = similarity.topk(1)
            top1 = indices[:, 0] == targets
            top1ACC = top1.sum() / len(top1)

            if top1ACC > best_seen_acc:
                best_seen_acc = top1ACC
                state_dict = model.state_dict()
                logging.info(f"Epoch {epoch} best_seen_acc: {best_seen_acc}")
                save_dict = {
                    'state_dict': state_dict,
                    'epoch': epoch
                }
                save_best_checkpoint_epoch(save_dict, is_best=True, gap=1,
                                           filename=os.path.join(config["model_path"],
                                                                 'checkpoint_epoch%d.pth.tar' % epoch),
                                           keep_all=True)
                config["best_model_path"] = os.path.join(config["model_path"], 'model_best_epoch%d.pth.tar' % epoch)

def extract_trn_feat(config, epoch, dataloader, text, model, device):
    actionlist, actiondict, actiontoken = text
    index_unknown = np.argwhere(actionlist == static_unknown_text)
    actionlist = np.delete(actionlist, index_unknown)

    save_path = config["extract_dir"]
    zsl_unseen_num = config["dataset_args"]["unseen_num"]
    dataset_name = config["dataset_args"]["dataset"]
    gpt_aug_actionlist = [GPT_AUG_DICT[dataset_name][word] for word in actionlist]
    if config["baseline_args"]["baseline"] == "bert":
        gpt_aug_actionlist = []

    model.eval()
    with torch.no_grad():
        all_targets = torch.zeros(0).to(device)
        all_vFeatures = torch.zeros(0).to(device)
        all_vis_lists = torch.zeros(0).to(device)
        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            vids, name, y_true, vis_type = sample
            if config["dataset_args"]["dataset"] == "USC":
                vids = vids.unsqueeze(1)

            vFeature, _, _ = model(vids.to(device), actionlist[:1], gpt_aug_actionlist[:1], type="all")

            target_batch = y_true.to(device)
            vis_type = vis_type.to(device)

            vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)

            all_targets = torch.cat([all_targets, target_batch], dim=0)
            all_vFeatures = torch.cat([all_vFeatures, vFeature], dim=0)
            all_vis_lists = torch.cat([all_vis_lists, vis_type], dim=0)

        known_trn_feat = all_vFeatures[all_vis_lists == 0]
        known_trn_targets = all_targets[all_vis_lists == 0]

        return known_trn_feat, known_trn_targets


def zsl_feat_extraction(config, train_feat, train_targets, val_mix_feat, val_mix_targets, val_mix_vis_lists, seen_t_feat, unseen_t_feat):
    print("######## ENABLE THE FEAT EXTRACTION ########")
    zsl_unseen_num = config["dataset_args"]["unseen_num"]
    save_path = config["extract_dir"]
    print(f"Feat save path: {save_path}")
    raw_save_path = config["extract_raw_dir"]

    unique_trn_targets = torch.unique(train_targets)
    seen_num = len(unique_trn_targets)
    trn_select_targets = unique_trn_targets[:len(unique_trn_targets) - zsl_unseen_num]
    val_select_targets = unique_trn_targets[len(unique_trn_targets) - zsl_unseen_num:]

    torch.save(train_feat, save_path + "/trn_feat.pth")
    torch.save(train_targets, save_path + "/trn_targets.pth")
    torch.save(unique_trn_targets, save_path + "/seen_targets.pth")
    torch.save(trn_select_targets, save_path + "/trn_select_targets.pth")
    torch.save(val_select_targets, save_path + "/val_select_targets.pth")


    all_seen_val_feat = val_mix_feat[val_mix_vis_lists == 0]
    all_unseen_val_feat = val_mix_feat[val_mix_vis_lists == 1]

    all_seen_val_targets = val_mix_targets[val_mix_vis_lists == 0]
    all_unseen_val_targets = val_mix_targets[val_mix_vis_lists == 1]

    all_unseen_val_targets = all_unseen_val_targets + seen_num

    unseen_unique_targets = torch.unique(all_unseen_val_targets)
    unseen_num = len(unseen_unique_targets)

    torch.save(all_seen_val_feat, save_path + "/test_seen_feat.pth")
    torch.save(all_unseen_val_feat, save_path + "/test_unseen_feat.pth")
    torch.save(all_seen_val_targets, save_path + "/test_seen_targets.pth")
    torch.save(all_unseen_val_targets, save_path + "/test_unseen_targets.pth")
    torch.save(unseen_unique_targets, save_path + "/unseen_targets.pth")


    all_t_feat = torch.cat((seen_t_feat, unseen_t_feat), dim=0)
    torch.save(seen_t_feat, save_path + "/seen_t_feat.pth")
    torch.save(unseen_t_feat, save_path + "/unseen_t_feat.pth")
    torch.save(all_t_feat, save_path + "/all_t_feat.pth")

    # torch.save(seen_t_feat, raw_save_path + "/seen_t_feat.pth")
    # torch.save(unseen_t_feat, raw_save_path + "/unseen_t_feat.pth")
    # torch.save(all_t_feat, raw_save_path + "/all_t_feat.pth")

    print("all features for zsl extracted and saved successfully!")