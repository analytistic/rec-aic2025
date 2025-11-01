import torch
from .gater import Gatelayer, BiasNoisyTopKGating
from .ffn import GLUFeedForward, PointWiseFeedForward
from .moe_component import Dispatcher


class MoeFFN(torch.nn.Module):
    def __init__(self, hidden_units, out_units, num_experts=4):
        super(MoeFFN, self).__init__()
        self.gater = Gatelayer(hidden_units, num_experts)
        self.experts = torch.nn.ModuleList([
            GLUFeedForward(hidden_units, out_units) for _ in range(num_experts)
        ])


    def forward(self, inputs):
        gates = self.gater(inputs)  # (batch_size, seq_len, num_experts)
        
        distribute_answer = torch.cat([expert(inputs).unsqueeze(-1) for expert in self.experts], -1)
        combine_answer = (distribute_answer * gates.unsqueeze(-2)).sum(dim=-1)  # (batch_size, seq_len, out_units )
        return combine_answer
    

class TopkMoeFFN(torch.nn.Module):
    def __init__(self, hidden_units, out_units, num_experts=4, top_k=2):
        super(TopkMoeFFN, self).__init__()
        self.gater = BiasNoisyTopKGating(hidden_units, num_experts, top_k=top_k)
        self.dispatcher = Dispatcher(num_experts, hidden_units, self.gater, top_k)

    def forward(self, inputs):
        output, load_F = self.dispatcher(inputs)
        return output

