import torch
from tqdm import tqdm
from data_utils.gpt_aug import GPT_AUG_DICT

def extract_trn_feat(config, dataloader, text, model, device):
    actionlist, actiondict, actiontoken = text

    dataset_name = config["dataset_args"]["dataset"]
    gpt_aug_actionlist = [GPT_AUG_DICT[dataset_name][word] for word in actionlist]

    model.eval()
    with torch.no_grad():
        all_targets = torch.zeros(0).to(device)
        all_vFeatures = torch.zeros(0).to(device)
        all_vis_lists = torch.zeros(0).to(device)
        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            vids, name, y_true, vis_type = sample

            vFeature, _ = model(vids.to(device), actionlist[:1], gpt_aug_actionlist[:1])

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
    zsl_unseen_num = config["dataset_args"]["unseen_num"]
    save_path = config["extract_dir"]

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

    torch.save(seen_t_feat, raw_save_path + "/seen_t_feat.pth")
    torch.save(unseen_t_feat, raw_save_path + "/unseen_t_feat.pth")
    torch.save(all_t_feat, raw_save_path + "/all_t_feat.pth")

    print("all features for zsl extracted and saved successfully!")