import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class RecommendLoss(nn.Module):
    """
    Supported loss types:

    """
    def __init__(self, cfg, loss_type: str="bce"):
        super(RecommendLoss, self).__init__()
        self.cfg = cfg
        self.loss_type = loss_type
        self.global_step = 0
        self.loss_map = {
            "bce" : self.bce_loss,
            "triplet": self.triplet_loss,
            "cosine_triplet": self.cosine_triplet_loss,
            "ado_infonce": self.ado_infonce,
        }

    def forward(self, log_feats, pos_embs, neg_embs, token_type, weights=None):
        loss, pos_score, neg_score, neg_var, neg_max = self.loss_map[self.loss_type](log_feats, pos_embs, neg_embs, token_type, self.cfg)
        return loss, pos_score, neg_score, neg_var, neg_max


    @staticmethod
    def bce_loss(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None, act_2_mask=None):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        pos_logits = torch.sum(log_feats * pos_embs, dim=-1)
        neg_logits = torch.sum(log_feats.unsqueeze(1) * neg_embs, dim=-1)
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        loss_func = nn.BCEWithLogitsLoss(reduction=cfg.bce.reduction)
        pos_loss = loss_func(pos_logits[pos_indices], pos_labels[pos_indices])
        neg_loss = loss_func(neg_logits[neg_indices], neg_labels[neg_indices])
        return cfg.bce.alpha * pos_loss + cfg.bce.beta * neg_loss, torch.mean(pos_logits[pos_indices]).item(), torch.mean(neg_logits[neg_indices]).item(), torch.var(neg_logits[neg_indices]).item(), torch.max(neg_logits[neg_indices]).item()

    @staticmethod
    def triplet_loss(log_feats, pos_embs, neg_embs, mask, cfg):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4

        
        pos_l2 = torch.norm(log_feats - pos_embs, dim=-1)
        neg_l2 = torch.norm(log_feats.unsqueeze(1) - neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(pos_l2[pos_indices] - neg_l2[neg_indices] + cfg.triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_l2[pos_indices]).item(), torch.mean(neg_l2[neg_indices]).item(), torch.var(neg_l2[neg_indices]).item(), torch.max(neg_l2[neg_indices]).item()


    @staticmethod
    def cosine_triplet_loss(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
            
        pos_cos = F.cosine_similarity(log_feats, pos_embs, dim=-1)
        neg_cos = F.cosine_similarity(log_feats.unsqueeze(1), neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(neg_cos[neg_indices] - pos_cos[pos_indices] + cfg.cosine_triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_cos[pos_indices]).item(), torch.mean(neg_cos[neg_indices]).item(), torch.var(neg_cos[neg_indices]).item(), torch.max(neg_cos[neg_indices]).item()



    @staticmethod
    def ado_infonce(log_feats, pos_embs, neg_embs, token_type, cfg):
        """
        ADO InfoNCE loss with in-batch negatives.
        """
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 3

        bs, seq_len, dim = log_feats.shape
        mask = (token_type == 1)


        log_feats = F.normalize(log_feats, p=2, dim=-1)  # Normalize sequence embeddings
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)

        pos_logits = F.cosine_similarity(log_feats, pos_embs, dim=-1)  # (bs, seq_len)
        pos_scores = pos_logits[(mask == 1)].mean().item()  # Average positive score

        neg_embs = F.normalize(neg_embs, p=2, dim=-1)  # Normalize negative embeddings
        neg_embs_all = neg_embs.reshape(-1, dim) # (bs*seq_len, dim)

        neg_logitys = torch.matmul(log_feats, neg_embs_all.transpose(-1, -2)) # (bs, seq_len, bs*seq_len)
        neg_scores = neg_logitys.mean().item()
        logtis = torch.cat([pos_logits.unsqueeze(-1), neg_logitys], dim=-1)  # bs, seq_len, neg_num+1
        

        if cfg.loss.weight_loss.act == True:
            logtis_0 = logtis[(mask == 1) & (act_0_mask == 1)] / cfg.ado_infonce.temperature  
            logtis_1 = logtis[(mask == 1) & (act_1_mask == 1)] / cfg.ado_infonce.temperature
            logtis_2 = logtis[(mask == 1) & (act_2_mask == 1)] / cfg.ado_infonce.temperature
            labels_0 = torch.zeros(logtis_0.shape[0], dtype=torch.long, device=logtis.device)
            labels_1 = torch.zeros(logtis_1.shape[0], dtype=torch.long, device=logtis.device)
            labels_2 = torch.zeros(logtis_2.shape[0], dtype=torch.long, device=logtis.device)
            loss_0 = F.cross_entropy(logtis_0, labels_0)
            loss_1 = F.cross_entropy(logtis_1, labels_1)
            loss_2 = F.cross_entropy(logtis_2, labels_2)
            loss = loss_0 * cfg.weight_loss.weight[1] + loss_1 * cfg.weight_loss.weight[0] + loss_2 * cfg.weight_loss.weight[2]
        else:
            logtis = logtis[(mask == 1)] / cfg.ado_infonce.temperature

            labels = torch.zeros(logtis.shape[0], dtype=torch.long, device=logtis.device)

            loss = F.cross_entropy(logtis, labels)


        return loss, pos_scores, neg_scores, neg_logitys.var().item(), neg_logitys.max().item()

    
