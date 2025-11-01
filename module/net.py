import torch
import torch.nn as nn

from layer import EmbeddingFusionGate, SeNet

class ItemNet(torch.nn.Module):
    def __init__(self, channels, hidden_units):
        super(ItemNet, self).__init__()
        self.feats_fusion_layer = SeNet(in_channels=channels,  ex_dim=2*channels, hidden_dim=hidden_units)
        self.fusion_layer = EmbeddingFusionGate(2*hidden_units, hidden_units)



    def forward(self, id_emb, feat_emb):
        x = self.feats_fusion_layer(feat_emb)
        x = self.fusion_layer(id_emb, x)


        
        return x
    

class UserNet(torch.nn.Module):
    def __init__(self, channels, hidden_units):
        super(UserNet, self).__init__()
        self.feats_fusion_layer = SeNet(in_channels=channels,  ex_dim=2*channels, hidden_dim=hidden_units)
        self.fusion_layer = EmbeddingFusionGate(2*hidden_units, hidden_units)



    def forward(self, id_emb, feat_emb):
        x = self.feats_fusion_layer(feat_emb)
        x = self.fusion_layer(id_emb, x)


        return x

