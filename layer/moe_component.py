
import torch
import torch.nn as nn
from .ffn import GLUFeedForward, PointWiseFeedForward




    


class Dispatcher(nn.Module):
    """
    实现稀疏分发，聚合
    """
    def __init__(self, num_experts, hidden_units, gater, top_k):
        super(Dispatcher, self).__init__()
        self.num_experts = num_experts
        self.gater = gater
        self.top_k = top_k
        self.hidden_units = hidden_units
        self.RMSnorm = nn.RMSNorm(hidden_units, eps=1e-8)

        nn.init.ones_(self.RMSnorm.weight)
        self.RMSnorm.weight.requires_grad = False

        self.experts = nn.ModuleList([
            GLUFeedForward(hidden_units, hidden_units) for _ in range(num_experts)
        ])

    def _dispatch(self, inputs, router):
        """
        inputs: (batch_size*seq_len, hidden_units)
        router: (batch_size*seq_len, num_experts) - 0/1指示每个样本选择的专家
        return:
            expert_inputs: list of (num_tokens_i, hidden_units)
            indices: list of (num_tokens_i,) - 原始输入的索引
        """
        expert_inputs = []
        indices = []
        for i in range(self.num_experts):
            mask = router[:, i]  # (batch_size*seq_len,)
            min_batch = inputs[mask==1]  # (num_tokens_i, hidden_units)
            expert_inputs.append(min_batch)
            indices.append(mask.nonzero(as_tuple=False).squeeze(-1))  # (num_tokens_i,)
        return expert_inputs, indices
    
    def _combine(self, expert_outputs, indices, batch_size, seq_len, router):
        """
        聚合
        """
        combine_outputs = torch.zeros(batch_size * seq_len, self.top_k, self.hidden_units,
                                    device=expert_outputs[0].device)
        
        # 预计算每个专家在每个token中的top_k位置
        cumsum_router = router.cumsum(dim=1) - 1  # (bs*len, num_experts)
        
        for expert_idx in range(self.num_experts):
            if expert_outputs[expert_idx] is None:
                continue
                
            expert_token_indices = indices[expert_idx]
            expert_output = expert_outputs[expert_idx]
            
            # 批量获取top_k位置
            k_positions = cumsum_router[expert_token_indices, expert_idx]
            
            # 批量赋值
            combine_outputs[expert_token_indices, k_positions] = expert_output
        
        return combine_outputs

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, hidden_units)
        bs, len, hidden_units = inputs.shape
        inputs = inputs.view(bs * len, hidden_units)

        gates, router, load_F = self.gater(inputs)  # (batch_size*seq_len, top_k), (batch_size*seq_len, num_experts), (num_experts,)

        norm_inputs = self.RMSnorm(inputs)
        expert_inputs, indices = self._dispatch(norm_inputs, router)  # list of (num_tokens_i, hidden_units), list of (num_tokens_i,)
        expert_outputs = [self.experts[i](expert_inputs[i]) if expert_inputs[i].size(0) > 0 else None for i in range(self.num_experts)]  # list of (num_tokens_i, hidden_units)
        answer = self._combine(expert_outputs, indices, bs, len, router)  
        answer = (answer * gates.unsqueeze(-1)).sum(dim=1)  # (batch_size*seq_len, hidden_units)
        answer = answer.view(bs, len, hidden_units)

        return answer, load_F


        

    

# if __name__ == "__main__":
#     from gater import BiasNoisyTopKGating
#     gater = BiasNoisyTopKGating(input_dim=16, num_experts=4, top_k=2)
#     dispatcher = Dispatcher(num_experts=4, hidden_units=16, gater=gater, top_k=2)
#     x = torch.randn(2, 3, 16)
#     out = dispatcher(x)
#     print(out.shape)  # (6, 2, 16)