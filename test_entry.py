import torch
from utils.open_set import val_parameter_define
from utils.validations import mix_validation
from utils.train_utils import extract_trn_feat
from utils.sup_model import SupModel

import warnings
warnings.filterwarnings("ignore")

def test_entry(config, dataloader, text, model, device):
    # *************** setup ***************#
    trnloader, val_tune_loader, val_mix_loader = dataloader
    epoch = 0
    open_set_df = config["args"]["open_set_df"]
    gzsl_metrics_df = config["args"]["gzsl_metrics_df"]

    # *************** load model ***************#
    checkpoint_path = config["args"]["test_model_path"]
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    specialist_model = None

    # TODO: add specialist model loader
    if config["args"]["local_model_path"] is not None:
        print("Local specialist model is loaded")
        specialist_model = SupModel(config, device)
        specialist_model_path = config["args"]["local_model_path"]
        specialist_model.load_state_dict(torch.load(specialist_model_path)['state_dict'])
        specialist_model.to(device)

    model.eval()
    model.clipmodel.eval()
    known_trn_vFeatures, known_trn_targets = extract_trn_feat(config, trnloader, text, model, device)
    if config["open_set_args"]["manual"] == False:
        # *************** define parameters for KNN using validation dataset *************** #
        auto_knn_params = val_parameter_define(config, val_tune_loader, text, model, known_trn_vFeatures,
                                               known_trn_targets, device)
    else:
        auto_knn_params = None

    eval_dict, val_feat_pack = mix_validation(config, epoch, val_mix_loader, text, model, known_trn_vFeatures,
                                              known_trn_targets, device, auto_knn_params, special_model=specialist_model)
    # [val_vFeatures, val_targets, val_vis_lists, pos_tFeature, neg_tFeature, gzsl_eval_dict] = val_feat_pack

    open_set_acc = eval_dict["open_set"]
    open_set_dict = eval_dict["open_set_dict"]

    os_precision = open_set_dict["precision"]
    os_recall = open_set_dict["recall"]
    os_fscore = open_set_dict["fscore"]

    gzsl_s = eval_dict["gzsl_dict"]["S"].detach().cpu().numpy()
    gzsl_u = eval_dict["gzsl_dict"]["U"].detach().cpu().numpy()
    gzsl_h = eval_dict["gzsl_dict"]["H"].detach().cpu().numpy()

    open_set_df.loc[len(open_set_df)] = [epoch, open_set_acc.detach().cpu().numpy(), 0, 0, 0, 0, os_precision,
                                         os_recall, os_fscore]

    gzsl_metrics_df.loc[len(gzsl_metrics_df)] = [gzsl_s, gzsl_u, gzsl_h]

    open_set_df.to_csv(config["model_path"] + f"/open_set_{config['baseline_args']['baseline']}_test.csv", index=False)
    gzsl_metrics_df.to_csv(config["model_path"] + f"/gzsl_metrics_{config['baseline_args']['baseline']}_test.csv", index=False)

    return
