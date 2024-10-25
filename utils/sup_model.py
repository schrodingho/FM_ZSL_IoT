import torch
import torch.nn as nn
import math

class SupModel(torch.nn.Module):
    def __init__(self, config, device):
        super(SupModel, self).__init__()
        self.device = device
        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.num_classes = config["dataset_args"]["seen_num"]
        self.feat_encoder = VisionTransformer(config["dataset_args"]["train_shape"], category=self.num_classes)

    def forward(self, vids):
        pred_out = self.feat_encoder(vids.float())
        return pred_out

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
        out = self.patch_merge(z)  # [batch, modal_leng, patch_num/2, output_dim]
        return out


class VisionTransformer(nn.Module):
    def __init__(self, train_shape, category, embedding_dim=256, patch_size=4, head_num=4, att_size=64):
        super().__init__()

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
