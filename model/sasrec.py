import torch
import torch.nn as nn
import torch.nn.functional as F
from module import ItemNet, UserNet, RecommendLoss, LogDecoder



class SASRec(nn.Module):
    def __init__(self, cfg, user_id_num, item_id_num, user_feat_dict, item_feat_dict):
        super(SASRec, self).__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(user_id_num+1, cfg.model.user_emb_dim)
        self.item_emb = nn.Embedding(item_id_num+1, cfg.model.item_emb_dim)
        self.user_feat_embs = nn.ModuleDict()
        for i, v in user_feat_dict.items():
            self.user_feat_embs[i] = nn.Embedding(v + 1, cfg.model.user_feat_emb_dim)
        self.item_feat_embs = nn.ModuleDict()
        for i, v in item_feat_dict.items():
            self.item_feat_embs[i] = nn.Embedding(v + 1, cfg.model.item_feat_emb_dim)

        self.item_net = ItemNet(channels=len(item_feat_dict), hidden_units=cfg.model.hidden_units)
        self.user_net = UserNet(channels=len(user_feat_dict), hidden_units=cfg.model.hidden_units)
        self.loss_func = RecommendLoss(cfg, loss_type=cfg.loss.type)
        self.decoder = LogDecoder(cfg.model)


    def feat2emb(self, input, include_user=True):
        
        if include_user:
            user_id, user_feat = input[2], input[3]
            user_emb = self.user_emb(user_id)
            user_feat_list = []

            for i, v in enumerate(self.user_feat_embs.items()):
                user_feat_list.append(v[1](user_feat[:, i]))

            user_feat_emb = torch.stack(user_feat_list, dim=-2)
            user_token = self.user_net(user_emb, user_feat_emb)
        
        id_seq, feat_seq = input[0], input[1]

        id_emb = self.item_emb(id_seq)
        item_feat_list = []
        for i, v in enumerate(self.item_feat_embs.items()):
            item_feat_list.append(v[1](feat_seq[..., i]))

        item_feat_emb = torch.stack(item_feat_list, dim=-2)

        tokens = self.item_net(id_emb, item_feat_emb)

        if include_user:
            j = input[4]
            bs, seq_len, _ = tokens.shape
            tokens[torch.arange(bs), j.clamp(min=0, max=seq_len - 1).long()] = user_token

        return tokens
    
    def log2embs(self, user_id, j, user_feat, id_seq, feat_seq, inter_time, act_type, token_type):
        tokens = self.feat2emb(input=[id_seq, feat_seq, user_id, user_feat, j], include_user=True)
        log_embs = self.decoder(tokens, j, inter_time, act_type, token_type)

        return log_embs


        

    

    def forward(self, 
                user_id,
                j,
                user_feat,
                id_seq,
                feat_seq,
                pos_seq,
                pos_feat,
                neg_seq,
                neg_feat,
                inter_time,
                act_type,
                token_type,
                ):
        
        log_embs = self.log2embs(user_id, j, user_feat, id_seq, feat_seq, inter_time, act_type, token_type)

        pos_embs = self.feat2emb(input=[pos_seq, pos_feat], include_user=False)
        neg_embs = self.feat2emb(input=[neg_seq, neg_feat], include_user=False)

        (
            mainloss,
            pos_score, 
            neg_score, 
            neg_var, 
            neg_max,
        ) = self.loss_func(
            log_embs,
            pos_embs,
            neg_embs,
            token_type,
        )

        return (
            mainloss,
            pos_score,
            neg_score,
            neg_var,
            neg_max,
        )
    

    def vali(self, 
            user_id,
            j,
            user_feat,
            id_seq,
            feat_seq,
            pos_seq,
            pos_feat,
            neg_seq,
            neg_feat,
            inter_time,
            act_type,
            token_type,
            ):
        
        log_embs = self.log2embs(user_id, j, user_feat, id_seq, feat_seq, inter_time, act_type, token_type)
        target_item = pos_seq[:, -1]
        log_embs = log_embs[:, -1, :]

        return log_embs, target_item
    
    def save_item_emb(self, item_id, item_feat):
        item_emb = self.feat2emb(input=[item_id, item_feat], include_user=False)
        return item_emb



        



                
            


       