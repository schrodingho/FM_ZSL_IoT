import torch
import argparse
# Additional Scripts
from train import TrainTestPipe

import os
from utils.AWADataset import clip_embedding

from config import cfg
import dill
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--g_cls_path', type=str, default=None)
parser.add_argument('--wgan_G_path', type=str, default=None)
parser.add_argument('--wgan_D_path', type=str, default=None)
parser.add_argument('--projection_path', type=str, default=None)
parser.add_argument('--g_fake_attr_path', type=str, default=None)
parser.add_argument('--g_fake_raw_path', type=str, default=None)
parser.add_argument('--dataset', type=str, default="USC")
parser.add_argument('--clip', type=int, default=1)

def main_pipeline(args):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'

    if args.dataset == 'USC':
        cfg.seen_class_number = 9
        cfg.unseen_class_number = 3
        cfg.x_dim = 128 * 6
        
    elif args.dataset == 'pamap':
        cfg.seen_class_number = 9
        cfg.unseen_class_number = 3
        cfg.x_dim = 171 * 36
    elif args.dataset == 'mmwave':
        cfg.seen_class_number = 22
        cfg.unseen_class_number = 5
        cfg.x_dim = 100 * 5
    elif args.dataset == 'wifi':
        cfg.seen_class_number = 22
        cfg.unseen_class_number = 5
        cfg.x_dim = 60 * 225
    else:
        raise ValueError("Dataset not supported!")

    if args.g_fake_raw_path:
        cfg.mypath = args.g_fake_raw_path
        cfg.clip = args.clip

    if args.g_fake_attr_path:
        cfg.mypath = args.g_fake_attr_path
        cfg.clip = args.clip
        cfg.x_dim = 1024

    print("Raw CLIP is used: ", cfg.clip)

    if args.train:
        args.g_cls_path = 'g_cls_model_1e4.pt'
        args.wgan_G_path = 'wgan_G_model_1e4.pt'
        args.wgan_D_path = 'wgan_D_model_1e4.pt'
        args.projection_path = 'projection_model_1e4.pt'
        delete_file(args.g_cls_path)
        delete_file(args.wgan_G_path)
        delete_file(args.wgan_D_path)

        ttp = TrainTestPipe(device, dataset='Widar')
        # ttp = TrainTestPipe(device)
        # ttp.load_model(args.g_cls_path, 'g_cls')
        # ttp.load_model([args.wgan_G_path, args.wgan_D_path], ['wgan_G', 'wgan_D'])
        # ttp.load_model(args.projection_path, 'projection')

        print('G_cls training process has been started!')
        ttp.train_g_cls()

        print('Wgan training process has been started!')
        ttp.train_wgan()

        print('Projection training process has been started!')
        ttp.train_projection()

        print('Test has been started!')
        ttp.test()
        save_res_path = "./clswagan_results"
        if not os.path.exists(save_res_path):
            os.makedirs(save_res_path)
        metrics_df_get = ttp.metrics_df
        metrics_df_get.to_csv(f"{save_res_path}/DCN_{args.dataset}.csv", index=False)

        if args.g_fake_raw_path:
            path = args.g_fake_raw_path
            generate_fake_raw_data(args, path, ttp)
        else:
            print("The fake raw data path does not exist!")
    else:
        args.g_cls_path = 'g_cls_model_1e4.pt'
        args.wgan_G_path = 'wgan_G_model_1e4.pt'
        args.wgan_D_path = 'wgan_D_model_1e4.pt'
        args.projection_path = 'projection_model_1e4.pt'
        # check if the path exists
        if not file_exist_check(args.g_cls_path) or not file_exist_check(args.wgan_G_path) or not file_exist_check(args.wgan_D_path) or not file_exist_check(args.projection_path):
            print("The pretrained model does not exist!")
            return

        ttp = TrainTestPipe(device, dataset='Widar')
        # ttp = TrainTestPipe(device)
        ttp.load_model(args.g_cls_path, 'g_cls')
        ttp.load_model([args.wgan_G_path, args.wgan_D_path], ['wgan_G', 'wgan_D'])
        ttp.load_model(args.projection_path, 'projection')

        if args.g_fake_raw_path:
            cfg.mypath = args.g_fake_raw_path
            path = args.g_fake_raw_path
            generate_fake_raw_data(args, path, ttp)
        else:
            print("The fake raw data path does not exist!")

def generate_fake_raw_data(cur_parser, path, ttp):
    print('Unseen fake raw data extraction process has been started!')
    seen_num = cfg.seen_class_number
    unseen_num = cfg.unseen_class_number

    # father_path = os.path.dirname(path)
    trn_targets = torch.load(path + "trn_targets.pth").detach().cpu().numpy().astype(int)
    each_class_num = (len(trn_targets) // seen_num)
    all_text_label_dict = dill.load(open(path + "all_text_label_dict.pkl", "rb"))
    all_label_text_dict = {v: k for k, v in all_text_label_dict.items()}
    if cfg.clip:
        # all_text_label_dict = dill.load(open(path + "all_text_label_dict.pkl", "rb"))
        # all_label_text_dict = {v: k for k, v in all_text_label_dict.items()}
        all_sentences = list(all_text_label_dict.keys())
        unseen_sentences = all_sentences[seen_num:]
        unseen_text_emb_feat = clip_embedding(unseen_sentences)
    else:
        unseen_text_emb_feat = torch.load(path + "unseen_t_feat.pth")
    index_list = [i for i in range(unseen_num)]
    index_tensor = torch.tensor(index_list).repeat(each_class_num)

    expand_unseen_text_emb_feat = unseen_text_emb_feat[index_tensor]
    unseen_fake_raw_data = ttp.unseen_fake_feat_extraction(expand_unseen_text_emb_feat)

    if cur_parser.dataset == 'USC':
        unseen_fake_raw_data = unseen_fake_raw_data.detach().cpu().numpy()
        print(unseen_fake_raw_data.shape)
        unseen_fake_raw_data = unseen_fake_raw_data.reshape((len(index_tensor), 128, 6))
        unseen_fake_targets = index_tensor.detach().cpu().numpy() + seen_num
        correspond_text = [all_label_text_dict[i] for i in unseen_fake_targets]
        dill.dump(correspond_text, open(cur_parser.g_fake_raw_path + "unseen_fake_text.pkl", "wb"))
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_raw_data.npy", unseen_fake_raw_data)
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_targets.npy", unseen_fake_targets)
    elif cur_parser.dataset == 'pamap':
        unseen_fake_raw_data = unseen_fake_raw_data.detach().cpu().numpy()
        print(unseen_fake_raw_data.shape)
        unseen_fake_raw_data = unseen_fake_raw_data.reshape((len(index_tensor), 171, 36))
        unseen_fake_targets = index_tensor.detach().cpu().numpy() + seen_num
        correspond_text = [all_label_text_dict[i] for i in unseen_fake_targets]
        dill.dump(correspond_text, open(cur_parser.g_fake_raw_path + "unseen_fake_text.pkl", "wb"))
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_raw_data.npy", unseen_fake_raw_data)
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_targets.npy", unseen_fake_targets)
    elif cur_parser.dataset == 'mmwave':
        unseen_fake_raw_data = unseen_fake_raw_data.detach().cpu().numpy()
        unseen_fake_raw_data = unseen_fake_raw_data.reshape((len(index_tensor), 100, 5))
        unseen_fake_targets = index_tensor.detach().cpu().numpy() + seen_num
        correspond_text = [all_label_text_dict[i] for i in unseen_fake_targets]
        dill.dump(correspond_text, open(cur_parser.g_fake_raw_path + "unseen_fake_text.pkl", "wb"))
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_raw_data.npy", unseen_fake_raw_data)
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_targets.npy", unseen_fake_targets)
    elif cur_parser.dataset == 'wifi':
        unseen_fake_raw_data = unseen_fake_raw_data.detach().cpu().numpy()
        unseen_fake_raw_data = unseen_fake_raw_data.reshape((len(index_tensor), 60, 225))
        unseen_fake_targets = index_tensor.detach().cpu().numpy() + seen_num
        correspond_text = [all_label_text_dict[i] for i in unseen_fake_targets]
        dill.dump(correspond_text, open(cur_parser.g_fake_raw_path + "unseen_fake_text.pkl", "wb"))
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_raw_data.npy", unseen_fake_raw_data)
        np.save(cur_parser.g_fake_raw_path + "unseen_fake_targets.npy", unseen_fake_targets)
    else:
        raise ValueError("Dataset not supported!")

    print("Unseen fake raw data has been saved!")
    return


def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} has been deleted.")
    else:
        print(f"{file_name} does not exist.")

def file_exist_check(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False

if __name__ == '__main__':
    args = parser.parse_args()
    main_pipeline(args)
