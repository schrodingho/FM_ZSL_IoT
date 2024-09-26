import torch
import torch.nn as nn
import clip
import numpy as np
import math
class CLIPrompt(torch.nn.Module):
    def __init__(self, config, text_all, device):
        super(CLIPrompt, self).__init__()
        self.device = device
        self.other_text = False
        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.hidden_size = 512
        self.feat_dim = 512

        self.clipmodel, _ = clip.load(config["args"]["backbone"], device=self.device,
                                      jit=False,
                                      return_intermediate_text_feature=0)

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        self.feat_encoder = VisionTransformer(config["dataset_args"]["train_shape"], category=self.hidden_size)

        self.prefix = config["args"]["prefix"]
        self.postfix = config["args"]["postfix"]

        # self.all_actionlist, self.all_actiondict, self.all_actiontoken, _, _, _, _, _, _ = text_all
        # _, _, _, self.seen_actionlist, self.seen_actiondict, self.seen_actiontoken, _, _, _ = text_all
        # _, _, _, _, _, _, self.unseen_actionlist, self.unseen_actiondict, self.unseen_actiontoken = text_all
        self.all_actionlist, self.all_actiondict, self.all_actiontoken = text_all
        self.actionlist = self.all_actionlist
        self.actiondict = self.all_actiondict
        self.actiontoken = self.all_actiontoken

        # self.actionlist = self.seen_actionlist
        # self.actiondict = self.seen_actiondict
        # self.actiontoken = self.seen_actiontoken

        self.tfm_layers = config["args"]["tfm_layers"]
        self.tfm_heads = config["args"]["tfm_heads"]

        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.initialize_parameters()

        self.configs = config

        self.cross_att = Cross_Att(self.hidden_size, self.hidden_size, self.hidden_size)

    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist), 77)


        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])
            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)


    def forward(self, vids, inp_actionlist, gpt_actionlist=None):
        vFeature = self.feat_encoder(vids.float())

        if gpt_actionlist != []:
            gpt_token = clip.tokenize(gpt_actionlist).to(self.device)
            # Hard Prompt
            tFeat_gpt = self.clipmodel.encode_text_original(gpt_token)

        self.replace_text_embedding(inp_actionlist)
        # Soft Prompt
        tFeature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken)

        if gpt_actionlist != []:
            tFeature = self.cross_att(q=tFeature, k=tFeat_gpt, v=tFeature)

        return vFeature, tFeature


class Cross_Att(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(Cross_Att, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, q, k, v):
        Q = self.q(q)  # Q: batch_size * seq_len * dim_k
        K = self.k(k)  # K: batch_size * seq_len * dim_k
        V = self.v(v)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(torch.mm(Q, K.T)) * self._norm_fact
        output = torch.mm(atten, V)

        return output





def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


"""
VisionTransformer Backbone from https://github.com/xushige/HAR-Dataset-Preprocess/blob/main/models/vit.py
"""

class VisionTransformerBlock(nn.Module):
    def __init__(self, input_dim=256, head_num=4, att_size=64):
        super().__init__()
        self.head_num = head_num
        self.att_size = att_size
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, head_num * att_size)
        self.key = nn.Linear(input_dim, head_num * att_size)
        self.value = nn.Linear(input_dim, head_num * att_size)
        self.att_mlp = nn.Sequential(
            nn.Linear(head_num * att_size, input_dim),
            nn.LayerNorm(input_dim)
        )
        self.downsample_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim)
        )
    def patch_merge(self, x):
        batch, modal_leng, patch_num, input_dim = x.shape
        if patch_num % 2:
            x = nn.ZeroPad2d((0, 0, 0, 1))(x)
        x0 = x[:, :, 0::2, :]  # [batch, modal_leng, patch_num / 2, input_dim]
        x1 = x[:, :, 1::2, :]  # # [batch, modal_leng, patch_num / 2, input_dim]
        x = torch.cat([x0, x1], dim=-1)  # [batch, modal_leng, patch_num / 2, input_dim * 2]
        x = nn.ReLU()(self.downsample_mlp(x))  # [batch, modal_leng, patch_num / 2, input_dim]
        return x

    def forward(self, x):
        '''
            x.shape: [batch, modal_leng, patch_num, input_dim]
        '''
        batch, modal_leng, patch_num, input_dim = x.shape
        # Q, K, V
        query = self.query(x).reshape(batch, modal_leng, patch_num, self.head_num, self.att_size).permute(0, 1, 3, 2,
                                                                                                          4)  # [batch, modal_leng, head_num, patch_num, att_size]
        key = self.key(x).reshape(batch, modal_leng, patch_num, self.head_num, self.att_size).permute(0, 1, 3, 4,
                                                                                                      2)  # [batch, modal_leng, head_num, att_size, patch_num]
        value = self.value(x).reshape(batch, modal_leng, patch_num, self.head_num, self.att_size).permute(0, 1, 3, 2,
                                                                                                          4)  # [batch, modal_leng, head_num, patch_num, att_size]
        # Multi Self-Attention Score
        z = torch.matmul(nn.Softmax(dim=-1)(torch.matmul(query, key) / (self.att_size ** 0.5)),
                         value)  # [batch, modal_leng, head_num, patch_num, att_size]
        z = z.permute(0, 1, 3, 2, 4).reshape(batch, modal_leng, patch_num,
                                             -1)  # [batch, modal_leng, patch_num, head_num*att_size]
        # Forward
        z = nn.ReLU()(x + self.att_mlp(z))  # [batch, modal_leng, patch_num, input_dim]
        out = self.patch_merge(z)  # 降采样[batch, modal_leng, patch_num/2, output_dim]
        return out


class VisionTransformer(nn.Module):
    def __init__(self, train_shape, category, embedding_dim=256, patch_size=4, head_num=4, att_size=64):
        super().__init__()
        # cut patch
        self.series_leng = train_shape[-2]
        self.modal_leng = train_shape[-1]
        self.patch_num = self.series_leng // patch_size

        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=(patch_size, 1),
            stride=(patch_size, 1),
            padding=0
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.modal_leng, self.patch_num, embedding_dim))
        # Multi Self-Attention Layer
        self.msa_layer = nn.Sequential(
            VisionTransformerBlock(embedding_dim, head_num, att_size),
            VisionTransformerBlock(embedding_dim, head_num, att_size),
            VisionTransformerBlock(embedding_dim, head_num, att_size)
        )
        # classification
        self.dense_tower = nn.Sequential(
            nn.Linear(self.modal_leng * math.ceil(self.patch_num / 8) * embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, category)
        )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.patch_conv(x)  # [batch, embedding_dim, patch_num, modal_leng]
        x = self.position_embedding + x.permute(0, 3, 2, 1)  # [batch, modal_leng, patch_num, embedding_dim]
        #    [batch, modal_leng, patch_num, input_dim]
        # -> [batch, modal_leng, patch_num/2, input_dim]
        # -> [batch, modal_leng, patch_num/4, input_dim]
        # -> [batch, modal_leng, patch_num/8, input_dim]
        x = self.msa_layer(x)
        x = nn.Flatten()(x)
        x = self.dense_tower(x)
        return x

