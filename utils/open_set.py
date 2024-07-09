from tqdm import tqdm
import torch
import numpy as np
from data_utils.gpt_aug import GPT_AUG_DICT

def val_parameter_define(config, dataloader, text, model, trn_vFeatures, trn_targets, device):
    actionlist, actiondict, actiontoken = text
    dataset_name = config["dataset_args"]["dataset"]
    gpt_aug_actionlist = [GPT_AUG_DICT[dataset_name][word] for word in actionlist]
    seen_num = config["dataset_args"]["seen_num"]
    if config["baseline_args"]["baseline"] == "bert":
        gpt_aug_actionlist = []

    model.eval()
    with torch.no_grad():
        targets = torch.zeros(0).to(device)
        vFeature_lists = torch.zeros(0).to(device)
        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            vids, name, y_true, vis_type = sample

            vFeature, _, _ = model(vids.to(device), actionlist[:1], gpt_aug_actionlist[:1], type="all")

            target_batch = y_true.to(device)
            vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)

            targets = torch.cat([targets, target_batch], dim=0)
            vFeature_lists = torch.cat([vFeature_lists, vFeature], dim=0)

    if config["open_set_args"]["cluster"] == True:
        targets_knn_tensor, targets_thres_tensor = knn_parameter_define(config, trn_vFeatures, trn_targets, vFeature_lists, targets)
        targets_knn_tensor = targets_knn_tensor.to(device)
        targets_thres_tensor = targets_thres_tensor.to(device)
        auto_knn_params = {"k": targets_knn_tensor, "v": targets_thres_tensor}
    else:
        targets_knn_val, target_thres = knn_parameter_define_2(config, trn_vFeatures, trn_targets, vFeature_lists, targets)
        auto_knn_params = {"k": targets_knn_val, "v": target_thres}

    return auto_knn_params

def knn_parameter_define(config, trn_vFeatures, trn_targets, val_vFeatures, val_targets):
    """
    define each cluster's threshold and k value
    """
    trn_unique_targets = torch.unique(trn_targets)
    val_unique_targets = torch.unique(val_targets)

    targets_thres_tensor = torch.zeros(len(trn_unique_targets))
    targets_thres_tensor = targets_thres_tensor.to(torch.float32)

    targets_knn_tensor = torch.zeros(len(trn_unique_targets))
    targets_knn_tensor = targets_knn_tensor.to(torch.long)

    k_defined_params = config["open_set_args"]["knn_percent"]
    percent_defined_params = config["open_set_args"]["cluster_percent"]
    for target in trn_unique_targets:
        target_idx = trn_targets == target
        target_trn_vFeatures = trn_vFeatures[target_idx]
        target_val_idx = val_targets == target
        target_val_vFeatures = val_vFeatures[target_val_idx]
        num_of_trn = target_trn_vFeatures.size(0)

        k_val = int(k_defined_params * num_of_trn)
        target_thres = os_detect_1_params_search(target_trn_vFeatures, target_val_vFeatures, k_val, percentage=percent_defined_params)

        idx = int(target.item())

        targets_knn_tensor[idx] = k_val
        targets_thres_tensor[idx] = target_thres

    return targets_knn_tensor, targets_thres_tensor


def knn_parameter_define_2(config, trn_vFeatures, trn_targets, val_vFeatures, val_targets):
    """
    globally defining the threshold (or k)
    """
    # k_defined_percent = config["open_set_args"]["knn_percent"]
    k_defined_val = config["open_set_args"]["knn_val"]
    percent_defined_params = config["open_set_args"]["cluster_percent"]

    # targets_knn_val define (use ratio or fixed value
    targets_knn_val = k_defined_val
    target_thres = os_detect_1_params_search(trn_vFeatures, val_vFeatures, targets_knn_val, percentage=percent_defined_params)

    # targets_knn_val = int(k_defined_percent * len(trn_targets))
    # target_thres = os_detect_1_params_search(trn_vFeatures, val_vFeatures, targets_knn_val, percentage=percent_defined_params)

    return targets_knn_val, target_thres

def os_detect_1(trn_vFeatures, val_vFeatures, knn_val, dis_threshold):
    # 1. depends on the training seen data (KNN-based)
    # calculate the distance between val_vFeatures and trn_vFeatures
    dist_wifi_text = torch.cdist(val_vFeatures.unsqueeze(0), trn_vFeatures.unsqueeze(0), p=2)
    dist_wifi_text = dist_wifi_text.squeeze(0)
    # find the k-th largest distance
    dist_vals, dist_inds = torch.topk(dist_wifi_text, knn_val, largest=False)
    dist_knn_vals = dist_vals[:, knn_val - 1]
    select_unseen_idx = dist_knn_vals >= dis_threshold
    select_seen_idx = dist_knn_vals < dis_threshold
    return select_seen_idx, select_unseen_idx

def os_detect_1_params_search(trn_vFeatures, val_vFeatures, knn_val, percentage=0.9):
    # 1. depends on the training seen data (KNN-based)
    # calculate the distance between val_vFeatures and trn_vFeatures
    dist_trn_val = torch.cdist(val_vFeatures.unsqueeze(0), trn_vFeatures.unsqueeze(0), p=2)
    dist_trn_val = dist_trn_val.squeeze(0)
    # find the k-th largest distance
    dist_vals, dist_inds = torch.topk(dist_trn_val, knn_val, largest=False)
    dist_knn_vals = dist_vals[:, knn_val - 1]

    # find a threshold to guarantee 90% of the dist_knn_vals <= threshold
    dist_knn_vals = dist_knn_vals.sort()[0]
    threshold_last = dist_knn_vals[:int(percentage * len(dist_knn_vals))][-1]

    return threshold_last

def os_detect_4(trn_vFeatures, val_vFeatures, trn_targets, val_targets, auto_knn_params):
    # 4. depends on the predefined threshold
    targets_knn_tensor = auto_knn_params["k"]
    target_thres_tensor = auto_knn_params["v"]

    unique_trn_targets = torch.unique(trn_targets)

    select_seen_idx = torch.zeros(len(val_vFeatures)).bool()
    # to device
    select_seen_idx = select_seen_idx.to(trn_vFeatures.device)
    select_seen_idx_val = torch.zeros(len(val_vFeatures))
    select_seen_idx_val = select_seen_idx_val.to(trn_vFeatures.device)

    for target in unique_trn_targets:
        target_idx = trn_targets == target
        target_trn_vFeatures = trn_vFeatures[target_idx]

        idx = int(target.item())

        k_val = targets_knn_tensor[idx]
        target_thres = target_thres_tensor[idx]

        dist_trn_val = torch.cdist(val_vFeatures.unsqueeze(0), target_trn_vFeatures.unsqueeze(0), p=2)
        dist_trn_val = dist_trn_val.squeeze(0)
        dist_vals, dist_inds = torch.topk(dist_trn_val, k_val, largest=False)
        dist_knn_vals = dist_vals[:, k_val - 1]
        # TODO: optimize the open-set recognition
        select_seen_idx = select_seen_idx | (dist_knn_vals <= target_thres)

        # select_seen_idx_val += (dist_knn_vals <= target_thres).int()
        # print(1)

    # select_seen_idx = select_seen_idx_val == 1
    select_unseen_idx = ~select_seen_idx

    return select_seen_idx, select_unseen_idx

def os_detect_4_text(seen_tFeatures, val_vFeatures, auto_knn_params):
    target_thres_tensor = auto_knn_params["v"]
    select_seen_idx = torch.zeros(len(val_vFeatures)).bool()
    select_seen_idx = select_seen_idx.to(seen_tFeatures.device)

    for target in range(len(seen_tFeatures)):
        select_tFeature = seen_tFeatures[target]
        target_thres = target_thres_tensor[target]
        # dist_val_tFeat = torch.cdist(val_vFeatures.unsqueeze(0), select_tFeature.unsqueeze(0), p=2)
        # dist_val_tFeat = dist_val_tFeat.squeeze(0)
        dist_val_tFeat = torch.norm(val_vFeatures - select_tFeature, dim=1)
        select_seen_idx = select_seen_idx | (dist_val_tFeat < target_thres)

    select_unseen_idx = ~select_seen_idx
    return select_seen_idx, select_unseen_idx

def os_detect_2(pos_tFeature, neg_tFeature, val_vFeatures):
    # 2. depends on the text embedding
    return

def os_detect_3(pos_tFeature, neg_tFeature, val_vFeatures):
    # 3. depends on the probability of product between text embedding and image embedding
    return