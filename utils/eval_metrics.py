import torch
import logging
from utils.open_set import os_detect_1, os_detect_4
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
def model_eval_metrics(config, epoch, trn_vFeatures, val_vFeatures, pos_tFeatures, neg_tFeatures, vis_type_lists, trn_targets, val_targets, auto_knn_params=None):
    logging.info(f"#####################################################\n"
                 f"Epoch: {epoch}")
    knn_val = config["open_set_args"]["knn_val"]
    dis_threshold = config["open_set_args"]["knn_threshold"]

    unique_seen_classes = torch.unique(val_targets[vis_type_lists == 0])
    unique_unseen_classes = torch.unique(val_targets[vis_type_lists == 1])

    trn_vFeatures = trn_vFeatures / trn_vFeatures.norm(dim=-1, keepdim=True)
    val_vFeatures = val_vFeatures / val_vFeatures.norm(dim=-1, keepdim=True)
    pos_tFeatures = pos_tFeatures / pos_tFeatures.norm(dim=-1, keepdim=True)
    neg_tFeatures = neg_tFeatures / neg_tFeatures.norm(dim=-1, keepdim=True)

    seen_idx = vis_type_lists == 0
    unseen_idx = vis_type_lists == 1

    eval_dict = {
        "open_set" : 0,
        "zsl_unseen_1": 0,
        "zsl_unseen_2": 0,
        "avg_recall": 0,
        "avg_precision": 0,
        "avg_f1": 0,
        "open_set_dict": {
            "FPR": 0,
            "FNR": 0,
            "TNR": 0,
            "TPR": 0,
            "precision": 0,
            "recall": 0,
            "fscore": 0,
        },
        "gzsl_dict": {
            "S": 0,
            "U": 0,
            "H": 0,
        }
    }

    if config["open_set_args"]["manual"] == False:
        if config["open_set_args"]["cluster"] == True:
            select_seen_idx, select_unseen_idx = os_detect_4(trn_vFeatures, val_vFeatures, trn_targets, val_targets, auto_knn_params)
        else:
            knn_val, dis_threshold = auto_knn_params["k"], auto_knn_params["v"]
            select_seen_idx, select_unseen_idx = os_detect_1(trn_vFeatures, val_vFeatures, knn_val, dis_threshold)
    else:
        select_seen_idx, select_unseen_idx = os_detect_1(trn_vFeatures, val_vFeatures, knn_val, dis_threshold)

    assert len(seen_idx) == len(select_seen_idx)
    assert len(unseen_idx) == len(select_unseen_idx)

    # seen_idx, unseen_idx
    # We treat seen as 1, unseen as 0
    y_true = unseen_idx.int().detach().cpu().numpy()
    y_pred = select_unseen_idx.int().detach().cpu().numpy()

    y_true_seen = seen_idx.int().detach().cpu().numpy()
    y_pred_seen = select_seen_idx.int().detach().cpu().numpy()
    os_seen_precision, os_seen_recall, os_seen_fscore, os_seen_support = precision_recall_fscore_support(y_true_seen, y_pred_seen, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    os_FPR = FP / (FP + TN)
    os_FNR = FN / (FN + TP)
    os_TNR = TN / (TN + FP)
    os_TPR = TP / (TP + FN)
    os_precision, os_recall, os_fscore, os_support = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"os_seen_precision: {os_seen_precision}, os_seen_recall: {os_seen_recall}, os_seen_fscore: {os_seen_fscore}")
    print(f"os_unseen_precision: {os_precision}, os_unseen_recall: {os_recall}, os_unseen_fscore: {os_fscore}")

    eval_dict["open_set_dict"]["FPR"] = os_FPR
    eval_dict["open_set_dict"]["FNR"] = os_FNR
    eval_dict["open_set_dict"]["TNR"] = os_TNR
    eval_dict["open_set_dict"]["TPR"] = os_TPR
    eval_dict["open_set_dict"]["precision"] = os_precision
    eval_dict["open_set_dict"]["recall"] = os_recall
    eval_dict["open_set_dict"]["fscore"] = os_fscore

    correct_hit_seen_idx = (seen_idx & (select_seen_idx == seen_idx)).bool()
    correct_hit_unseen_idx = (unseen_idx & (select_unseen_idx == unseen_idx)).bool()
    incorrect_hit_seen_idx = (seen_idx & (select_unseen_idx == seen_idx)).bool()
    incorrect_hit_unseen_idx = (unseen_idx & (select_seen_idx == unseen_idx)).bool()

    correct_hit_seen_vFeature = val_vFeatures[correct_hit_seen_idx]
    correct_hit_unseen_vFeature = val_vFeatures[correct_hit_unseen_idx]
    # TODO: fix bug

    # seen is classified as unseen
    incorrect_hit_seen_vFeature = val_vFeatures[incorrect_hit_seen_idx]

    correct_hit_seen_targets = val_targets[correct_hit_seen_idx]
    correct_hit_unseen_targets = val_targets[correct_hit_unseen_idx]

    seen_logits = correct_hit_seen_vFeature @ pos_tFeatures.t()
    unseen_logits = correct_hit_unseen_vFeature @ neg_tFeatures.t()

    incorrect_unseen_logits = incorrect_hit_seen_vFeature @ neg_tFeatures.t()


    seen_sim = seen_logits.softmax(dim=-1)
    unseen_sim = unseen_logits.softmax(dim=-1)

    incorrect_unseen_sim = incorrect_unseen_logits.softmax(dim=-1)

    seen_max_idx = seen_sim.argmax(dim=-1)
    unseen_max_idx = unseen_sim.argmax(dim=-1)

    incorrect_seen_in_unseen_target = incorrect_unseen_sim.argmax(dim=-1)

    # essential results: zsl classification results
    zsl_seen_hits_idx = seen_max_idx == correct_hit_seen_targets
    zsl_unseen_hits_idx = unseen_max_idx == correct_hit_unseen_targets

    ##################### new evaluation metrics ######################
    zsl_unseen_hits_target = unseen_max_idx[zsl_unseen_hits_idx]
    # incorrect_seen_to_unseen_max_idx: incorrect_unseen_max_idx
    val_unseen_targets = val_targets[unseen_idx]
    val_seen_targets = val_targets[seen_idx]
    # TODO: correct_hit_unseen_targets existing potential problems (seen also contain 0 -> 2 label)
    per_class_acc_unseen = per_class_acc_calc(unseen_max_idx, correct_hit_unseen_targets, val_unseen_targets, unique_unseen_classes)
    per_class_acc_seen = per_class_acc_calc(seen_max_idx, correct_hit_seen_targets, val_seen_targets, unique_seen_classes)

    harmonic_mean = 2 * per_class_acc_seen * per_class_acc_unseen / (per_class_acc_seen + per_class_acc_unseen)

    val_unseen_target_dict, unique_targets = target_cnt(val_unseen_targets)
    zsl_unseen_hits_target_dict, _ = target_cnt(zsl_unseen_hits_target, other=True, unique_targets=unique_targets)
    incorrect_seen_in_unseen_target_dict, _ = target_cnt(incorrect_seen_in_unseen_target, other=True, unique_targets=unique_targets)
    unseen_pred_dict, _ = target_cnt(unseen_max_idx, other=True, unique_targets=unique_targets)


    #### TODO: confusion matrix
    #### evaluation metrics
    avg_recall, avg_precision, avg_f1 = avg_metrics(val_unseen_target_dict, zsl_unseen_hits_target_dict, unseen_pred_dict, incorrect_seen_in_unseen_target_dict)

    # metric 1:
    seen_hits_rate = correct_hit_seen_idx.sum() / seen_idx.sum()
    unseen_hits_rate = correct_hit_unseen_idx.sum() / unseen_idx.sum()

    print(f"[Seen] correct_seen_hits / all_seen: {seen_hits_rate}\n"
          f"[Unseen] correct_unseen_hits / all_unseen: {unseen_hits_rate}\n")
    print(f"Open Set Acc: {(seen_hits_rate + unseen_hits_rate) / 2}")
    logging.info(f"[Seen] correct_seen_hits / all_seen: {seen_hits_rate}\n"
                 f"[Unseen] correct_unseen_hits / all_unseen: {unseen_hits_rate}\n"
                 f"Open Set Acc: {(seen_hits_rate + unseen_hits_rate) / 2}")


    # metric 2:
    zsl_seen_hits_div_correct_seen = zsl_seen_hits_idx.sum() / len(correct_hit_seen_targets)
    zsl_unseen_hits_div_correct_unseen = zsl_unseen_hits_idx.sum() / len(correct_hit_unseen_targets)
    print(f"[Seen] seen_zsl_hits / correct_seen_hits: {zsl_seen_hits_div_correct_seen}\n"
          f"[Unseen] unseen_zsl_hits / correct_unseen_hits: {zsl_unseen_hits_div_correct_unseen}")

    logging.info(f"[Seen] seen_zsl_hits / correct_seen_hits: {zsl_seen_hits_div_correct_seen}\n"
                    f"[Unseen] unseen_zsl_hits / correct_unseen_hits: {zsl_unseen_hits_div_correct_unseen}")


    # metric 3:
    zsl_seen_hits_div_select_seen = zsl_seen_hits_idx.sum() / select_seen_idx.sum()
    zsl_unseen_hits_div_select_unseen = zsl_unseen_hits_idx.sum() / select_unseen_idx.sum()


    # metric 4:
    zsl_seen_hits_div_seen_idx = zsl_seen_hits_idx.sum() / seen_idx.sum()
    zsl_unseen_hits_div_unseen_idx = zsl_unseen_hits_idx.sum() / unseen_idx.sum()

    print(f"[Seen] ZSL_seen_hits / all_seen: {zsl_seen_hits_div_seen_idx}\n"
          f"[Unseen] ZSL_unseen_hits / all_unseen: {zsl_unseen_hits_div_unseen_idx}")

    logging.info(f"[Seen] ZSL_seen_hits / all_seen: {zsl_seen_hits_div_seen_idx}\n"
                    f"[Unseen] ZSL_unseen_hits / all_unseen: {zsl_unseen_hits_div_unseen_idx}")



    print(f"[Seen] Per class acc: {per_class_acc_seen}")
    print(f"[Unseen] Per class acc: {per_class_acc_unseen}")
    print(f"Harmonic mean: {harmonic_mean}")
    logging.info(f"[Seen] Per class acc: {per_class_acc_seen}\n"
                    f"[Unseen] Per class acc: {per_class_acc_unseen}\n"
                    f"Harmonic mean: {harmonic_mean}")
    eval_dict["gzsl_dict"]["S"] = per_class_acc_seen
    eval_dict["gzsl_dict"]["U"] = per_class_acc_unseen
    eval_dict["gzsl_dict"]["H"] = harmonic_mean

    eval_dict["open_set"] = (seen_hits_rate + unseen_hits_rate) / 2
    eval_dict["zsl_unseen_1"] = zsl_unseen_hits_div_select_unseen
    # eval_dict["zsl_unseen_2"] = zsl_unseen_hits_div_unseen_idx
    # modify to per class acc
    eval_dict["zsl_unseen_2"] = per_class_acc_unseen
    eval_dict["avg_recall"] = avg_recall
    eval_dict["avg_precision"] = avg_precision
    eval_dict["avg_f1"] = avg_f1

    return eval_dict


def target_cnt(targets, other=False, unique_targets=None):
    target_dict = {}
    if not other:
        unique_targets = torch.unique(targets).detach().cpu().numpy()
        unique_targets = unique_targets.astype(int)
    for target in unique_targets:
        target_dict[target] = 0
    for target in unique_targets:
        target_dict[target] = (targets == target).sum().item()

    return target_dict, unique_targets

def avg_metrics(all_unseen_targets_dict, corr_zsl_unseen_targets_dict, all_unseen_pred_dict, incorr_seen_in_unseen_targets_dict):
    all_targets = list(all_unseen_targets_dict.keys())
    target_recall_dict = {}
    target_precision_dict = {}
    target_f1_dict = {}
    for target in all_targets:
        target_recall_dict[target] = 0
        target_precision_dict[target] = 0
        target_f1_dict[target] = 0
    avg_recall = 0
    avg_precision = 0
    avg_f1 = 0
    total_num_of_targets = 0
    for target in all_targets:
        if all_unseen_pred_dict[target] + incorr_seen_in_unseen_targets_dict[target] == 0:
            precision_target = 0
        else:
            precision_target = corr_zsl_unseen_targets_dict[target] / (all_unseen_pred_dict[target] + incorr_seen_in_unseen_targets_dict[target])
        target_precision_dict[target] = precision_target
        
        if all_unseen_targets_dict[target] == 0:
            recall_target = 0
        else:
            recall_target = corr_zsl_unseen_targets_dict[target] / all_unseen_targets_dict[target]
        target_recall_dict[target] = recall_target

        if precision_target == 0 and recall_target == 0:
            f1_target = 0
        else:
            f1_target = 2 * (precision_target * recall_target) / (precision_target + recall_target)
        target_f1_dict[target] = f1_target

        total_num_of_targets += all_unseen_targets_dict[target]

    for target in all_targets:
        avg_recall += target_recall_dict[target] * all_unseen_targets_dict[target] / total_num_of_targets
        avg_precision += target_precision_dict[target] * all_unseen_targets_dict[target] / total_num_of_targets
        avg_f1 += target_f1_dict[target] * all_unseen_targets_dict[target] / total_num_of_targets

    return avg_recall, avg_precision, avg_f1


def per_class_acc_calc(select_unseen_pred, select_unseen_targets, all_unseen_targets, unique_unseen_classes):
    per_class_acc = torch.zeros(unique_unseen_classes.shape)
    for i in unique_unseen_classes:
        is_class = all_unseen_targets == i
        cur_class_target_idx = select_unseen_targets == i
        correct_class = select_unseen_pred[cur_class_target_idx] == i
        per_class_acc[i.long()] = correct_class.sum() / is_class.sum()
    return torch.mean(per_class_acc)

def per_class_acc_calc2(select_unseen_pred, select_unseen_targets, all_unseen_targets, unique_unseen_classes, seen_num=None):
    per_class_acc = torch.zeros(unique_unseen_classes.shape)
    for i in unique_unseen_classes:
        is_class = all_unseen_targets == i
        cur_class_target_idx = select_unseen_targets == i
        correct_class = select_unseen_pred[cur_class_target_idx] == i
        if seen_num is not None:
            per_class_acc[i.long() - seen_num] = correct_class.sum() / is_class.sum()
        else:
            per_class_acc[i.long()] = correct_class.sum() / is_class.sum()
    return torch.mean(per_class_acc)

def gzsl_metrics(config, epoch, trn_vFeatures, val_vFeatures, pos_tFeatures, neg_tFeatures, vis_type_lists, trn_targets, val_targets, auto_knn_params=None):
    val_vFeatures = val_vFeatures / val_vFeatures.norm(dim=-1, keepdim=True)
    pos_tFeatures = pos_tFeatures / pos_tFeatures.norm(dim=-1, keepdim=True)
    neg_tFeatures = neg_tFeatures / neg_tFeatures.norm(dim=-1, keepdim=True)

    seen_idx = vis_type_lists == 0
    unseen_idx = vis_type_lists == 1

    eval_dict = {
        "open_set" : 0,
        "zsl_unseen_1": 0,
        "zsl_unseen_2": 0,
        "avg_recall": 0,
        "avg_precision": 0,
        "avg_f1": 0,
        "open_set_dict": {
            "FPR": 0,
            "FNR": 0,
            "TNR": 0,
            "TPR": 0,
            "precision": 0,
            "recall": 0,
            "fscore": 0,
        },
        "gzsl_dict": {
            "S": 0,
            "U": 0,
            "H": 0,
        }
    }

    seen_num = config["dataset_args"]["seen_num"]

    val_seen_feat = val_vFeatures[seen_idx]
    val_seen_targets = val_targets[seen_idx]

    val_unseen_feat = val_vFeatures[unseen_idx]
    val_unseen_targets = val_targets[unseen_idx] + seen_num

    unique_seen_classes = torch.unique(val_seen_targets)
    unique_unseen_classes = torch.unique(val_unseen_targets)

    all_t_Feat = torch.cat((pos_tFeatures, neg_tFeatures), dim=0)

    seen_logits = val_seen_feat @ all_t_Feat.t()
    unseen_logits = val_unseen_feat @ all_t_Feat.t()

    seen_sim = seen_logits.softmax(dim=-1)
    unseen_sim = unseen_logits.softmax(dim=-1)

    seen_max_idx = seen_sim.argmax(dim=-1)
    unseen_max_idx = unseen_sim.argmax(dim=-1)

    zsl_seen_hits_idx = seen_max_idx == val_seen_targets
    zsl_unseen_hits_idx = unseen_max_idx == val_unseen_targets

    ###################################################################
    zsl_unseen_hits_target = unseen_max_idx[zsl_unseen_hits_idx]

    ##################### new evaluation metrics ######################
    # TODO: finish this
    seen_per_class_acc = per_class_acc_calc2(seen_max_idx, val_seen_targets, val_seen_targets, unique_seen_classes)
    unseen_per_class_acc = per_class_acc_calc2(unseen_max_idx, val_unseen_targets, val_unseen_targets, unique_unseen_classes, seen_num=seen_num)

    H = 2 * seen_per_class_acc * unseen_per_class_acc / (seen_per_class_acc + unseen_per_class_acc)


    print(f"[X Open][Seen] per class acc: {seen_per_class_acc}")
    print(f"[X Open][Unseen] per class acc: {unseen_per_class_acc}")
    print(f"[X Open][H] harmonic mean: {H}")
    logging.info(f"[X Open][Seen] per class acc: {seen_per_class_acc}\n"
                 f"[X Open][Unseen] per class acc: {unseen_per_class_acc}\n"
                    f"[X Open][H] harmonic mean: {H}")

    eval_dict["gzsl_dict"]["S"] = seen_per_class_acc
    eval_dict["gzsl_dict"]["U"] = unseen_per_class_acc
    eval_dict["gzsl_dict"]["H"] = H

    val_unseen_target_dict, unique_unseen_targets = target_cnt(val_unseen_targets)
    zsl_unseen_hits_target_dict, _ = target_cnt(zsl_unseen_hits_target, other=True, unique_targets=unique_unseen_targets)
    unseen_pred_dict, _ = target_cnt(unseen_max_idx, other=True, unique_targets=unique_unseen_targets)
    incorrect_seen_in_unseen_target_dict, _ = target_cnt(seen_max_idx, other=True, unique_targets=unique_unseen_targets)

    avg_recall, avg_precision, avg_f1 = avg_metrics(val_unseen_target_dict, zsl_unseen_hits_target_dict, unseen_pred_dict, incorrect_seen_in_unseen_target_dict)

    eval_dict["zsl_unseen_2"] = unseen_per_class_acc
    eval_dict["avg_recall"] = avg_recall
    eval_dict["avg_precision"] = avg_precision
    eval_dict["avg_f1"] = avg_f1

    return eval_dict


def special_model_eval_metrics(special_model, val_mix_loader, config, epoch, trn_vFeatures, val_vFeatures, pos_tFeatures, neg_tFeatures, vis_type_lists, trn_targets, val_targets, device, auto_knn_params=None):
    print("== specialist model eval metrics start ==")
    knn_val = config["open_set_args"]["knn_val"]
    dis_threshold = config["open_set_args"]["knn_threshold"]

    unique_seen_classes = torch.unique(val_targets[vis_type_lists == 0])
    unique_unseen_classes = torch.unique(val_targets[vis_type_lists == 1])


    trn_vFeatures = trn_vFeatures / trn_vFeatures.norm(dim=-1, keepdim=True)
    val_vFeatures = val_vFeatures / val_vFeatures.norm(dim=-1, keepdim=True)
    pos_tFeatures = pos_tFeatures / pos_tFeatures.norm(dim=-1, keepdim=True)
    neg_tFeatures = neg_tFeatures / neg_tFeatures.norm(dim=-1, keepdim=True)

    seen_idx = vis_type_lists == 0
    unseen_idx = vis_type_lists == 1

    eval_dict = {
        "open_set" : 0,
        "zsl_unseen_1": 0,
        "zsl_unseen_2": 0,
        "avg_recall": 0,
        "avg_precision": 0,
        "avg_f1": 0,
        "open_set_dict": {
            "FPR": 0,
            "FNR": 0,
            "TNR": 0,
            "TPR": 0,
            "precision": 0,
            "recall": 0,
            "fscore": 0,
        },
        "gzsl_dict": {
            "S": 0,
            "U": 0,
            "H": 0,
        }
    }
    if config["open_set_args"]["manual"] == False:
        if config["open_set_args"]["cluster"] == True:
            select_seen_idx, select_unseen_idx = os_detect_4(trn_vFeatures, val_vFeatures, trn_targets, val_targets, auto_knn_params)
        else:
            knn_val, dis_threshold = auto_knn_params["k"], auto_knn_params["v"]
            select_seen_idx, select_unseen_idx = os_detect_1(trn_vFeatures, val_vFeatures, knn_val, dis_threshold)
    else:
        select_seen_idx, select_unseen_idx = os_detect_1(trn_vFeatures, val_vFeatures, knn_val, dis_threshold)

    assert len(seen_idx) == len(select_seen_idx)
    assert len(unseen_idx) == len(select_unseen_idx)

    correct_hit_seen_idx = (seen_idx & (select_seen_idx == seen_idx)).bool()
    correct_hit_unseen_idx = (unseen_idx & (select_unseen_idx == unseen_idx)).bool()
    incorrect_hit_seen_idx = (seen_idx & (select_unseen_idx == seen_idx)).bool()
    incorrect_hit_unseen_idx = (unseen_idx & (select_seen_idx == unseen_idx)).bool()

    correct_hit_seen_vFeature = val_vFeatures[correct_hit_seen_idx]
    correct_hit_unseen_vFeature = val_vFeatures[correct_hit_unseen_idx]
    # TODO: fix bug

    # selected_seen_raw_data = raw_data_lists[correct_hit_seen_idx]
    special_model.eval()
    with torch.no_grad():
        special_all_y_pred = torch.zeros(0).to(device)
        for idx, sample in tqdm(enumerate(val_mix_loader), total=len(val_mix_loader)):
            vids, _, _, _ = sample
            if config["dataset_args"]["dataset"] == "USC":
                vids = vids.unsqueeze(1)

            pred, _ = special_model(vids.to(device))

            _, predicted = torch.max(pred, 1)
            special_all_y_pred = torch.cat((special_all_y_pred, predicted), dim=0)


    special_all_y_pred = special_all_y_pred[correct_hit_seen_idx]
    #     batch_raw_data = selected_seen_raw_data[i:i + tmp_batch_size]
    #     batch_raw_data = [item for sublist in batch_raw_data for item in sublist]
    #     batch_raw_data = torch.stack(batch_raw_data)
    #     batch_raw_data = batch_raw_data.to(special_model.device)
    #     batch_raw_data = special_model.clipmodel.encode_text_original(batch_raw_data)
    #     if i == 0:
    #         seen_logits = correct_hit_seen_vFeature @ batch_raw_data.t()
    #     else:
    #         seen_logits = torch.cat((seen_logits, correct_hit_seen_vFeature @ batch_raw_data.t()), dim=0)


    # seen is classified as unseen
    incorrect_hit_seen_vFeature = val_vFeatures[incorrect_hit_seen_idx]

    correct_hit_seen_targets = val_targets[correct_hit_seen_idx]
    correct_hit_unseen_targets = val_targets[correct_hit_unseen_idx]

    seen_logits = correct_hit_seen_vFeature @ pos_tFeatures.t()
    unseen_logits = correct_hit_unseen_vFeature @ neg_tFeatures.t()

    incorrect_unseen_logits = incorrect_hit_seen_vFeature @ neg_tFeatures.t()

    seen_sim = seen_logits.softmax(dim=-1)
    unseen_sim = unseen_logits.softmax(dim=-1)

    incorrect_unseen_sim = incorrect_unseen_logits.softmax(dim=-1)

    seen_max_idx = seen_sim.argmax(dim=-1)
    unseen_max_idx = unseen_sim.argmax(dim=-1)

    incorrect_seen_in_unseen_target = incorrect_unseen_sim.argmax(dim=-1)

    # essential results: zsl classification results
    zsl_seen_hits_idx = seen_max_idx == correct_hit_seen_targets
    zsl_unseen_hits_idx = unseen_max_idx == correct_hit_unseen_targets

    ##################### new evaluation metrics ######################
    zsl_unseen_hits_target = unseen_max_idx[zsl_unseen_hits_idx]
    # incorrect_seen_to_unseen_max_idx: incorrect_unseen_max_idx
    val_unseen_targets = val_targets[unseen_idx]
    val_seen_targets = val_targets[seen_idx]
    # TODO: correct_hit_unseen_targets existing potential problems (seen also contain 0 -> 2 label)
    per_class_acc_unseen = per_class_acc_calc(unseen_max_idx, correct_hit_unseen_targets, val_unseen_targets, unique_unseen_classes)
    per_class_acc_seen = per_class_acc_calc(seen_max_idx, correct_hit_seen_targets, val_seen_targets, unique_seen_classes)

    per_class_speical_seen = per_class_acc_calc(special_all_y_pred, correct_hit_seen_targets, val_seen_targets, unique_seen_classes)

    harmonic_mean = 2 * per_class_acc_seen * per_class_acc_unseen / (per_class_acc_seen + per_class_acc_unseen)
    harmonic_mean_special = 2 * per_class_speical_seen * per_class_acc_unseen / (per_class_speical_seen + per_class_acc_unseen)

    val_unseen_target_dict, unique_targets = target_cnt(val_unseen_targets)
    zsl_unseen_hits_target_dict, _ = target_cnt(zsl_unseen_hits_target, other=True, unique_targets=unique_targets)
    incorrect_seen_in_unseen_target_dict, _ = target_cnt(incorrect_seen_in_unseen_target, other=True, unique_targets=unique_targets)
    unseen_pred_dict, _ = target_cnt(unseen_max_idx, other=True, unique_targets=unique_targets)


    #### TODO: confusion matrix
    #### evaluation metrics
    avg_recall, avg_precision, avg_f1 = avg_metrics(val_unseen_target_dict, zsl_unseen_hits_target_dict, unseen_pred_dict, incorrect_seen_in_unseen_target_dict)

    # metric 1:
    seen_hits_rate = correct_hit_seen_idx.sum() / seen_idx.sum()
    unseen_hits_rate = correct_hit_unseen_idx.sum() / unseen_idx.sum()

    print(f"[Seen] correct_seen_hits / all_seen: {seen_hits_rate}\n"
          f"[Unseen] correct_unseen_hits / all_unseen: {unseen_hits_rate}\n")
    print(f"Open Set Acc: {(seen_hits_rate + unseen_hits_rate) / 2}")
    logging.info(f"[Seen] correct_seen_hits / all_seen: {seen_hits_rate}\n"
                 f"[Unseen] correct_unseen_hits / all_unseen: {unseen_hits_rate}\n"
                 f"Open Set Acc: {(seen_hits_rate + unseen_hits_rate) / 2}")


    # metric 2:
    zsl_seen_hits_div_correct_seen = zsl_seen_hits_idx.sum() / len(correct_hit_seen_targets)
    zsl_unseen_hits_div_correct_unseen = zsl_unseen_hits_idx.sum() / len(correct_hit_unseen_targets)
    print(f"[Seen] seen_zsl_hits / correct_seen_hits: {zsl_seen_hits_div_correct_seen}\n"
          f"[Unseen] unseen_zsl_hits / correct_unseen_hits: {zsl_unseen_hits_div_correct_unseen}")

    logging.info(f"[Seen] seen_zsl_hits / correct_seen_hits: {zsl_seen_hits_div_correct_seen}\n"
                    f"[Unseen] unseen_zsl_hits / correct_unseen_hits: {zsl_unseen_hits_div_correct_unseen}")


    # metric 3:
    zsl_seen_hits_div_select_seen = zsl_seen_hits_idx.sum() / select_seen_idx.sum()
    zsl_unseen_hits_div_select_unseen = zsl_unseen_hits_idx.sum() / select_unseen_idx.sum()

    # print(f"[Seen] ZSL_seen_hits / select_seen: {zsl_seen_hits_div_select_seen}\n"
    #       f"[Unseen] ZSL_unseen_hits / select_unseen: {zsl_unseen_hits_div_select_unseen}")
    #
    # logging.info(f"[Seen] ZSL_seen_hits / select_seen: {zsl_seen_hits_div_select_seen}\n"
    #                 f"[Unseen] ZSL_unseen_hits / select_unseen: {zsl_unseen_hits_div_select_unseen}")

    # metric 4:
    zsl_seen_hits_div_seen_idx = zsl_seen_hits_idx.sum() / seen_idx.sum()
    zsl_unseen_hits_div_unseen_idx = zsl_unseen_hits_idx.sum() / unseen_idx.sum()

    print(f"[Seen] ZSL_seen_hits / all_seen: {zsl_seen_hits_div_seen_idx}\n"
          f"[Unseen] ZSL_unseen_hits / all_unseen: {zsl_unseen_hits_div_unseen_idx}")

    logging.info(f"[Seen] ZSL_seen_hits / all_seen: {zsl_seen_hits_div_seen_idx}\n"
                    f"[Unseen] ZSL_unseen_hits / all_unseen: {zsl_unseen_hits_div_unseen_idx}")

    # print(f"[Unseen] Per class acc: {per_class_acc_unseen}")
    # logging.info(f"[Unseen] Per class acc: {per_class_acc_unseen}")

    print(f"[Seen] Per class acc: {per_class_acc_seen}")
    print(f"[Seen] Special model per class acc: {per_class_speical_seen}")
    print(f"[Unseen] Per class acc: {per_class_acc_unseen}")
    print(f"Harmonic mean: {harmonic_mean}")
    print(f"Harmonic mean special: {harmonic_mean_special}")
    logging.info(f"[Seen] Per class acc: {per_class_acc_seen}\n"
                    f"[Unseen] Per class acc: {per_class_acc_unseen}\n"
                    f"Harmonic mean: {harmonic_mean}\n"
                    f"Harmonic mean special: {harmonic_mean_special}")
    eval_dict["gzsl_dict"]["S"] = per_class_acc_seen
    eval_dict["gzsl_dict"]["U"] = per_class_acc_unseen
    eval_dict["gzsl_dict"]["H"] = harmonic_mean

    eval_dict["open_set"] = (seen_hits_rate + unseen_hits_rate) / 2
    eval_dict["zsl_unseen_1"] = zsl_unseen_hits_div_select_unseen
    # eval_dict["zsl_unseen_2"] = zsl_unseen_hits_div_unseen_idx
    # modify to per class acc
    eval_dict["zsl_unseen_2"] = per_class_acc_unseen
    eval_dict["avg_recall"] = avg_recall
    eval_dict["avg_precision"] = avg_precision
    eval_dict["avg_f1"] = avg_f1

    return eval_dict

# unit test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.randint(0, 10, (100,)).to(device)
    target_dict = target_cnt(targets)
    print(target_dict)