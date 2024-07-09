import time
import os
import torch
import random
import numpy as np
from utils.other import delete_redundant_epoch_dirs, save_unseen_best
from utils.loss_func import SupConLoss
import logging
from utils.open_set import val_parameter_define
from utils.validations import mix_validation
from tqdm import tqdm
from data_utils.gpt_aug import GPT_AUG_DICT

import warnings
warnings.filterwarnings("ignore")

def train_CLIPrompt(config, dataloader, text, model, optimizer, lr_scheduler, device):
    temperature = 0.2
    logging.info(
        f"KNN_VAL: {config['open_set_args']['knn_val']}, "
        f"KNN_THRESHOLD: {config['open_set_args']['knn_threshold']}"
    )

    loss_supcon = SupConLoss(temperature=temperature)

    timestart = time.time()
    iteration = config["args"]["start_iter"]
    numContrast = config["args"]["numContrast"]
    if config["dataset_args"]["fake"]:
        fakeloader, trnloader, val_tune_loader, val_mix_loader = dataloader
        iteration_epoch = len(fakeloader)
        train_loader = fakeloader
    else:
        trnloader, val_tune_loader, val_mix_loader = dataloader
        iteration_epoch = len(trnloader)
        train_loader = trnloader
    actionlist, actiondict, actiontoken = text
    dataset = config["dataset_args"]["dataset"]
    seen_num = config["dataset_args"]["seen_num"]

    model.train()
    model.clipmodel.eval()
    best_open_acc = 0
    best_unseen_acc_1 = 0
    best_H = 0
    epoch = 0
    epochs = config["args"]["epochs"]
    open_set_df = config["args"]["open_set_df"]
    metrics_df = config["args"]["metrics_df"]
    no_open_metrics_df = config["args"]["no_open_metrics_df"]
    gzsl_metrics_df = config["args"]["gzsl_metrics_df"]
    gzsl_no_open_metrics_df = config["args"]["gzsl_no_open_metrics_df"]

    while epoch < epochs:
        for idx, sample in enumerate(train_loader):
            vids, name, _, _ = sample
            uniqname = np.unique(name)
            numNeg = numContrast - len(uniqname)
            complement = list(set(actionlist) - set(uniqname))
            inp_actionlist = uniqname.tolist() + random.sample(complement, min(numNeg, len(complement)))
            gpt_aug_actionlist = [GPT_AUG_DICT[dataset][word] for word in inp_actionlist]
            targets = torch.tensor([inp_actionlist.index(n) for n in name]).to(device)
            vFeature, tFeature, _ = model(vids.to(device), inp_actionlist, gpt_aug_actionlist, type="all")

            if not config["dataset_args"]["fake"]:
                tFeature = tFeature[:seen_num, :]

            vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
            tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)

            logits = vFeature @ tFeature.t() / 0.07

            targets_list = targets.tolist()
            positive_t = tFeature[targets_list]

            sup_v_Feature = vFeature.unsqueeze(1)
            sup_v_Feature = torch.cat((sup_v_Feature, positive_t.unsqueeze(1)), dim=1)

            loss = loss_supcon(sup_v_Feature, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if iteration % 100 == 0:
                similarity = logits.softmax(dim=-1)
                values, indices = similarity.topk(1)
                top1 = indices[:, 0] == targets
                top1ACC = top1.sum() / len(top1)

                print('dataset: {},'.format(dataset),
                      'epoch[{}][{}/{}]'.format(iteration // iteration_epoch, idx, iteration_epoch),
                      'epoch[{}], '.format(epoch),
                      'time {:.01f}s,'.format(time.time() - timestart),
                      'loss {:.03f},'.format(loss.detach().cpu().numpy()),
                      'top1 {:.03f},'.format(top1ACC.detach().cpu().numpy()),
                      )
                timestart = time.time()

            iteration += 1

        if epoch >= config["args"]["val_epoch"]:
            # TODO: 1. seen validation -> acc -> best_acc -> saving
            # val_model_seen(config, epoch, val_tune_loader, text, model, device)
            # TODO: 2. parameter tuning (training data sampling?)
            # if epoch >= 1 and (epoch + 1) % 4 == 0:
            model.eval()
            start_val_epoch = 0
            if config["dataset_args"]["dataset"] == "wifi":
                start_val_epoch = 10
            if epoch >= start_val_epoch:
                known_trn_vFeatures, known_trn_targets = extract_trn_feat(config, trnloader, text, model, device)
                if config["open_set_args"]["manual"] == False:
                    auto_knn_params = val_parameter_define(config, val_tune_loader, text, model, known_trn_vFeatures, known_trn_targets, device)
                else:
                    auto_knn_params = None
                # TODO: 3. open_set recognition, ZSL -> best_open_acc -> saving
                eval_dict, val_feat_pack = mix_validation(config, epoch, val_mix_loader, text, model, known_trn_vFeatures, known_trn_targets, device, auto_knn_params)
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

                if open_set > best_open_acc:
                    best_open_acc = open_set
                    print(f"Epoch: {epoch}, Best Open Set Acc: {best_open_acc}")
                    logging.info(f"Epoch: {epoch}, Best Open Set Acc: {best_open_acc}")

                # sum_acc = metric_seen_acc + metric_unseen_acc
                if zsl_unseen_1 > best_unseen_acc_1:
                    best_unseen_acc_1 = zsl_unseen_1
                    print(f"Epoch: {epoch}, Best Unseen Acc (select): {best_unseen_acc_1}")
                    logging.info(f"Epoch: {epoch}, Best Unseen Acc (select): {best_unseen_acc_1}")

                if gzsl_h > best_H:
                    best_H = gzsl_h
                    # print(f"Epoch: {epoch}, Best Unseen Acc (all): {best_unseen_acc_2}")
                    # logging.info(f"Epoch: {epoch}, Best Unseen Acc (all): {best_unseen_acc_2}")
                    print(f"Epoch: {epoch}, Best H: {best_H}")
                    logging.info(f"Epoch: {epoch}, Best H: {best_H}")

                    best_epoch = epoch
                    best_iteration = iteration

                    # save_best_checkpoint_epoch(save_dict, is_best=True, gap=1,
                    #                      filename=os.path.join(config["model_path"],
                    #                                            'checkpoint_epoch%d.pth.tar' % epoch),
                    #                      keep_all=True)
                    if config["baseline_args"]["ablation"] == "nol":
                        print("saving noL checkpoint")
                        state_dict = model.state_dict()
                        save_dict = {
                            'state_dict': state_dict,
                            'optimizer': optimizer.state_dict(),
                            'iteration': iteration,
                            'epoch': epoch
                        }
                        save_unseen_best(save_dict, filename=os.path.join(config["model_path"], 'model_best.pth.tar'))

                    if config["baseline_args"]["baseline"] == 0 and config["args"]["save"]:
                        print("saving best checkpoint")
                        state_dict = model.state_dict()

                        # saving the best checkpoint trn_features, val_features, pos_tFeature, neg_tFeature, vis_lists, targets
                        save_feat_dir = f"{config['model_path']}/epoch{epoch}"
                        os.makedirs(save_feat_dir, exist_ok=True)

                        torch.save(known_trn_vFeatures, f"{save_feat_dir}/vFeatures_train.pt")
                        torch.save(known_trn_targets, f"{save_feat_dir}/targets_train.pt")
                        zsl_feat_extraction(config, known_trn_vFeatures, known_trn_targets, val_vFeatures,
                                            val_targets, val_vis_lists, pos_tFeature, neg_tFeature)

                        torch.save(val_vFeatures, f"{save_feat_dir}/vFeatures_val_mix.pt")
                        torch.save(val_targets, f"{save_feat_dir}/targets_val_mix.pt")
                        torch.save(val_vis_lists, f"{save_feat_dir}/vis_val_mix.pt")
                        torch.save(pos_tFeature, f"{save_feat_dir}/pos_tFeature_train.pt")
                        torch.save(neg_tFeature, f"{save_feat_dir}/neg_tFeature_val_unseen.pt")

                        delete_redundant_epoch_dirs(config["model_path"])

                        save_dict = {
                            'state_dict': state_dict,
                            'optimizer': optimizer.state_dict(),
                            'iteration': iteration,
                            'epoch': epoch
                        }
                        save_unseen_best(save_dict, filename=os.path.join(config["model_path"], 'model_best.pth.tar'))


            model.train()
            model.clipmodel.eval()

        epoch += 1

    open_set_df.to_csv(config["model_path"] + f"/open_set_{config['baseline_args']['baseline']}.csv", index=False)
    metrics_df.to_csv(config["model_path"] + f"/metrics_{config['baseline_args']['baseline']}.csv", index=False)
    no_open_metrics_df.to_csv(config["model_path"] + f"/no_open_metrics_{config['baseline_args']['baseline']}.csv", index=False)

    gzsl_metrics_df.to_csv(config["model_path"] + f"/gzsl_metrics_{config['baseline_args']['baseline']}.csv", index=False)
    gzsl_no_open_metrics_df.to_csv(config["model_path"] + f"/gzsl_no_open_metrics_{config['baseline_args']['baseline']}.csv", index=False)


    return best_open_acc

def extract_trn_feat(config, dataloader, text, model, device):
    actionlist, actiondict, actiontoken = text

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