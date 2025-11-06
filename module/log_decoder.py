import torch
import torch.nn as nn
from .former import HSTUEncoder
from layer import EmbeddingFusionGate, SeNet

class CrossFeatFusion(torch.nn.Module):
    def __init__(self, cat_dim, hidden_units):
        super(CrossFeatFusion, self).__init__()
        self.gate = EmbeddingFusionGate(cat_emb_dim=cat_dim, fusion_dim=hidden_units)

    def forward(self, user_emb, item_emb):
        user_emb = user_emb.expand(-1, item_emb.shape[1], -1) 
        fusion_emb = self.gate(user_emb, item_emb)
        return fusion_emb


class LogDecoder(nn.Module):
    def __init__(self, cfg):
        super(LogDecoder, self).__init__()
        self.cfg = cfg




        self.time_stamp_emb = torch.nn.ModuleDict(
            {
                "hour": torch.nn.Embedding(25, cfg.hidden_units, padding_idx=0),
                "day": torch.nn.Embedding(32, cfg.hidden_units, padding_idx=0),
                "month": torch.nn.Embedding(13, cfg.hidden_units, padding_idx=0),
                "minute": torch.nn.Embedding(61, cfg.hidden_units, padding_idx=0),
            }
        )

        self.act_emb = torch.nn.Embedding(3, cfg.hidden_units, padding_idx=0)

        self.next_act_emb = torch.nn.Embedding(3, cfg.hidden_units, padding_idx=0)


        self.emb_dropout = torch.nn.Dropout(p=cfg.dropout_rate)
        self.encoder = HSTUEncoder(cfg.encoder)
        self.act_fusion = CrossFeatFusion(2*cfg.hidden_units, cfg.hidden_units)


        self.time_fusion = SeNet(1+2, 10, cfg.hidden_units)


    def forward(self, tokens, j, inter_time, act_type, token_type, renew_seq):
        """
        Args:
            id_seqs: 序列ID
            feat_seqs: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            seq_time: 序列时间特征
            seq_action_type: 序列动作类型
            scale: 缩放因子，用于缩放输入的ID和特征序列


        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size, maxlen, dim = tokens.shape
        scale = dim ** 0.5
        tokens *= scale


        pad_mask = (inter_time == 0)

        # hour = ((inter_time // 3600) % 24 + 1).masked_fill(pad_mask, 0).to(tokens.device)
        day = ((inter_time // 86400) % 31 + 1).masked_fill(pad_mask, 0).to(tokens.device)
        month = ((inter_time // (86400 * 30)) % 12 + 1).masked_fill(pad_mask, 0).to(tokens.device)
        # minute = ((inter_time // 60) % 60 + 1).masked_fill(pad_mask, 0).to(tokens.device)

        diff_matrix = torch.log(torch.abs(inter_time[:, :, None] - inter_time[:, None, :]) + 1)
        diff_matrix = torch.floor(diff_matrix).to(torch.int64)
        diff_matrix = torch.clamp(diff_matrix, 0, self.cfg.timediff_buckets-1)
        diff_matrix[torch.arange(batch_size), :, j.clamp(min=0, max=maxlen - 1).long()]=0

        # positions = torch.arange(maxlen, device=tokens.device).unsqueeze(0).expand(batch_size, -1)  # (bs, maxlen)
        # pos_diff_matrix = positions[:, :, None] - positions[:, None, :]  # (bs, maxlen, maxlen)
        # pos_diff_matrix = torch.abs(pos_diff_matrix)  # 取绝对值
        # pos_diff_matrix = torch.clamp(pos_diff_matrix, 0, self.cfg.pos_buckets-1)  # 限制范围
        # pos_diff_matrix[torch.arange(batch_size), :, j.clamp(min=0, max=maxlen - 1).long()] = 0  
        # indices_matrix = torch.searchsorted(self.time_diff_percentiles, diff_matrix, right=True)
        # diff_matrix = torch.where(diff_matrix == len(self.time_diff_percentiles), len(self.time_diff_percentiles) - 1, indices_matrix)

        # hour_emb = self.time_stamp_emb["hour"](hour)
        day_emb = self.time_stamp_emb["day"](day)
        month_emb = self.time_stamp_emb["month"](month)
        # minute_emb = self.time_stamp_emb["minute"](minute)


        time = torch.stack([day_emb, month_emb], dim=-2)
        tokens = self.time_fusion(torch.cat([tokens.unsqueeze(2), time], dim=-2))


        act_emb = self.act_emb(act_type.to(tokens.device))
        tokens = self.act_fusion(tokens, act_emb)

        tokens = self.emb_dropout(tokens) 

        mask = token_type
        


        log_embs = self.encoder(tokens, mask, diff_matrix = diff_matrix.to(tokens.device), renew_seq = renew_seq)  # (bs, seq_len, dim)

        return log_embs
