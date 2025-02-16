import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn import preprocessing
import numpy as np
# Additional Scripts
from config import cfg
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import dill
import clip

def clip_embedding(sentences):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


class AWADataset(Dataset):
    res_mat = None
    atts_mat = None

    # def __init__(self, set):
    #     super().__init__()
    #
    #     loc = self.atts_mat[set].squeeze() - 1
    #
    #     self.features = torch.from_numpy(self.res_mat['features'][..., loc]).float().T
    #     self.atts = torch.from_numpy(self.atts_mat['att']).float().T
    #     self.labels = torch.from_numpy((self.res_mat['labels'] - 1)[loc]).long()
    #
    # def __getitem__(self, idx):
    #     return {'feature': self.features[idx, :],
    #             'label': self.labels[idx],
    #             'attribute': self.atts[self.labels[idx][0]]}
    #
    # def __len__(self):
    #     return self.labels.shape[0]


class MyDataset(Dataset):

    # res_mat = sio.loadmat(cfg.res_path)
    # atts_mat = sio.loadmat(cfg.atts_path)
    def __init__(self, set):
        super().__init__()
        path = cfg.mypath
        print()
        trn_feat = torch.load(path + "/trn_feat.pth").detach().cpu().numpy()
        # normalize
        trn_feat = preprocessing.normalize(trn_feat, norm='l2')
        # to int
        trn_targets = torch.load(path + "trn_targets.pth").detach().cpu().numpy().astype(int)
        trn_select_classes = torch.load(path + "trn_select_targets.pth").detach().cpu().numpy().astype(int)
        val_select_classes = torch.load(path + "val_select_targets.pth").detach().cpu().numpy().astype(int)
        unique_seen_classes = torch.load(path + "seen_targets.pth").detach().cpu().numpy().astype(int)


        test_seen_feat = torch.load(path + "test_seen_feat.pth").detach().cpu().numpy()
        test_seen_feat = preprocessing.normalize(test_seen_feat, norm='l2')

        test_unseen_feat = torch.load(path + "test_unseen_feat.pth").detach().cpu().numpy()
        test_unseen_feat = preprocessing.normalize(test_unseen_feat, norm='l2')

        test_seen_targets = torch.load(path + "test_seen_targets.pth").detach().cpu().numpy().astype(int)
        test_unseen_targets = torch.load(path + "test_unseen_targets.pth").detach().cpu().numpy().astype(int)
        unique_unseen_classes = torch.load(path + "unseen_targets.pth").detach().cpu().numpy().astype(int)

        X_trainval_gzsl = trn_feat.T
        X_test_seen = test_seen_feat.T
        X_test_unseen = test_unseen_feat.T

        labels_trainval_gzsl = trn_targets
        labels_test_seen = test_seen_targets
        labels_test_unseen = test_unseen_targets
        labels_test = np.concatenate((labels_test_seen, labels_test_unseen), axis=0)

        train_classes = trn_select_classes
        val_classes = val_select_classes
        trainval_classes_seen = unique_seen_classes
        test_classes_seen = unique_seen_classes
        test_classes_unseen = unique_unseen_classes
        test_classes = np.concatenate((unique_seen_classes, unique_unseen_classes), axis=0)

        path = cfg.mypath
        sig = torch.load(path + "all_t_feat.pth").detach().cpu().numpy()
        sig = preprocessing.normalize(sig, norm='l2')
        sig = sig.T
        trainval_sig = sig[:, trainval_classes_seen - 1]
        train_sig = sig[:, train_classes - 1]
        val_sig = sig[:, val_classes - 1]
        test_sig = sig[:, test_classes - 1]  # Entire Signature Matrix

        if set == "trainval_loc":
            self.features = torch.from_numpy(X_trainval_gzsl).float().T
            self.atts = torch.from_numpy(sig).float().T
            self.labels = torch.from_numpy(labels_trainval_gzsl).long().unsqueeze(1)
        elif set == "test_seen_loc":
            self.features = torch.from_numpy(X_test_seen).float().T
            self.atts = torch.from_numpy(sig).float().T
            self.labels = torch.from_numpy(labels_test_seen).long().unsqueeze(1)
        elif set == "test_unseen_loc":
            self.features = torch.from_numpy(X_test_unseen).float().T
            self.atts = torch.from_numpy(sig).float().T
            self.labels = torch.from_numpy(labels_test_unseen).long().unsqueeze(1)
        # loc = self.atts_mat[set].squeeze() - 1
        #
        # self.features = torch.from_numpy(self.res_mat['features'][..., loc]).float().T
        # self.atts = torch.from_numpy(self.atts_mat['att']).float().T
        # self.labels = torch.from_numpy((self.res_mat['labels'] - 1)[loc]).long()

    def __getitem__(self, idx):
        return {'feature': self.features[idx, :],
                'label': self.labels[idx],
                'attribute': self.atts[self.labels[idx][0]]}

    def __len__(self):
        return self.labels.shape[0]