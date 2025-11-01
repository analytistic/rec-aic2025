import torch

from layer import HSTUBlock, MoeFFN



class HSTUEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.ffn_norm = torch.nn.ModuleList()
        self.last_norm = torch.nn.RMSNorm(cfg.hidden_units, eps=1e-8)
        for _ in range(cfg.num_blocks):
            self.layers.append(HSTUBlock(cfg.hidden_units, cfg.num_heads))
            self.norm_layers.append(torch.nn.RMSNorm(cfg.hidden_units, eps=1e-8))
            self.ffn_layers.append(MoeFFN(cfg.hidden_units, cfg.hidden_units, num_experts=cfg.num_experts))
            self.ffn_norm.append(torch.nn.RMSNorm(cfg.hidden_units, eps=1e-8))

        # self.drop_out = torch.nn.Dropout(p=cfg.ffn_dropout_rate)

    def forward(self, q, mask=None, diff_matrix=None):

        ones_matrix = torch.ones((q.shape[1], q.shape[1]), device=q.device, dtype=torch.bool)
        attention_mask_tril = torch.tril(ones_matrix)

        attention_mask_pad = (mask != 0).to(q.device)

        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        for i, layer in enumerate(self.layers):
            q_norm = self.norm_layers[i](q)
            q = layer(q_norm, mask=attention_mask, diff_matrix=diff_matrix) + q
            q = q + self.ffn_layers[i](self.ffn_norm[i](q))

        q = self.last_norm(q)
        return q