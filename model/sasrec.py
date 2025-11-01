import torch
import torch.nn as nn
import torch.nn.functional as F
from module import ItemNet, UserNet


class SASRec(nn.Module):
    def __init__(self, cfg, user_id_num, item_id_num, user_feat_dict, item_feat_dict):
        super(SASRec, self).__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(user_id_num, cfg.model.user_emb_dim)
        self.item_emb = nn.Embedding(item_id_num, cfg.model.item_emb_dim)
        self.user_feat_embs = nn.ModuleDict()
        for i, v in user_feat_dict.items():
            self.user_feat_embs[i] = nn.Embedding(v + 1, cfg.model.user_feat_emb_dim)
        self.item_feat_embs = nn.ModuleDict()
        for i, v in item_feat_dict.items():
            self.item_feat_embs[i] = nn.Embedding(v + 1, cfg.model.item_feat_emb_dim)

        self.item_net = ItemNet(cfg)
        self.useer_net = UserNet(cfg)


    def feat2tensor(self, input, include_user=True):
        

       