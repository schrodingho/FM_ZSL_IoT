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
from utils.train_utils import extract_trn_feat, zsl_feat_extraction

import warnings
warnings.filterwarnings("ignore")

def train_entry(config, dataloader, text, model, optimizer, lr_scheduler, device):
    temperature = 0.2
    logit_scale = 0.07

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
            vFeature, tFeature = model(vids.to(device), inp_actionlist, gpt_aug_actionlist)

            if not config["dataset_args"]["fake"]:
                tFeature = tFeature[:seen_num, :]

            vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
            tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)

            logits = vFeature @ tFeature.t() / logit_scale

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

