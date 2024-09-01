from tqdm import tqdm
import torch
from utils.eval_metrics import model_eval_metrics, gzsl_metrics, special_model_eval_metrics
from data_utils.gpt_aug import GPT_AUG_DICT

def mix_validation(config, epoch, dataloader, text, model, trn_vFeatures, trn_targets, device, auto_knn_params=None, special_model=None):
    actionlist, actiondict, actiontoken = text
    dataset_name = config["dataset_args"]["dataset"]
    gpt_aug_actionlist = [GPT_AUG_DICT[dataset_name][word] for word in actionlist]
    seen_num = config["dataset_args"]["seen_num"]

    model.eval()
    with torch.no_grad():
        similarity, val_targets = torch.zeros(0).to(device), torch.zeros(0).to(device)
        vis_lists = torch.zeros(0).to(device)
        vFeature_lists = torch.zeros(0).to(device)

        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            vids, name, y_true, vis_type = sample
            vis_type = vis_type.to(device)

            if idx == 0:
                vFeature, tFeature = model(vids.to(device), actionlist, gpt_aug_actionlist)
                all_tFeature = tFeature
                pos_tFeature = tFeature[:seen_num, :]
                neg_tFeature = tFeature[seen_num:, :]
                pos_tFeature = pos_tFeature / pos_tFeature.norm(dim=-1, keepdim=True)
                neg_tFeature = neg_tFeature / neg_tFeature.norm(dim=-1, keepdim=True)
            else:
                vFeature, _ = model(vids.to(device), actionlist[:1], gpt_aug_actionlist[:1])

            target_batch = y_true.to(device)

            vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)

            val_targets = torch.cat([val_targets, target_batch], dim=0)
            vFeature_lists = torch.cat([vFeature_lists, vFeature], dim=0)
            vis_lists = torch.cat([vis_lists, vis_type], dim=0)

    if not special_model:
        eval_dict = model_eval_metrics(config, epoch, trn_vFeatures, vFeature_lists, pos_tFeature, neg_tFeature, vis_lists, trn_targets, val_targets, auto_knn_params=auto_knn_params)
    else:
        eval_dict = special_model_eval_metrics(special_model, dataloader, config, epoch,
                                               trn_vFeatures, vFeature_lists, pos_tFeature, neg_tFeature,
                                               vis_lists, trn_targets, val_targets, device=device, auto_knn_params=auto_knn_params)
    # TODO: remove this?
    # gzsl_eval_dict = gzsl_metrics(config, epoch, trn_vFeatures, vFeature_lists, pos_tFeature, neg_tFeature, vis_lists, trn_targets, val_targets, auto_knn_params=auto_knn_params)
    gzsl_eval_dict = None
    # TODO: modify the parameter return
    mix_val_data_pack = [vFeature_lists, val_targets, vis_lists, all_tFeature, neg_tFeature, gzsl_eval_dict]

    return eval_dict, mix_val_data_pack
