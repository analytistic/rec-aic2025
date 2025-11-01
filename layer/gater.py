
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gatelayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Gatelayer, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        gates = torch.nn.Softmax(dim=-1)(self.W(x))
        return gates
    




class BiasNoisyTopKGating(nn.Module):
    """带噪声的Top-K偏置路由门控网络
    input -> clean_logits + noise -> gates
    使用loss_free : https://kexue.fm/archives/10757
    bias梯度更新公式: b <- b - \alpha * (load_F - Q) / RMS(load_F - Q), 其中Q是离散均匀分布

    gates: 打分 = sigmoid(clean_logits + noise)
    router: 偏置路由 = top_k(gates + bias)
    load_F: 负载分布 = sum[(gating > 0) * 1/k]

    input_dim: 输入维度
    num_experts: 专家数量
    top_k: 选择前K个专家
    noise_epsilon: 噪声强度

    """
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-2, alpha=0.001):
        super(BiasNoisyTopKGating, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.alpha = alpha

        

        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts)).requires_grad_(False)

        nn.init.normal_(self.w_gate.weight, 0.0, 0.1)
        nn.init.normal_(self.w_noise.weight, 0.0, 0.1)
        nn.init.constant_(self.bias, 0.01)
        
        self.load_F = 0


    def _router_to_F(self, router):
        """计算每个专家的负载分布
        router: (batch_size, num_experts) - 0/1指示每个样本选择的专家
        f_i = 1/k if i in top-k(router) else 0
        F_i = E[f_i]
        """
        load_F = torch.mean(router * (1.0 / self.top_k), dim=0) # (num_experts,)
        return load_F
    
    def _bias_grad_hook(self):
        with torch.no_grad():
            Q = (1.0 / self.num_experts) * torch.ones(self.num_experts, device=self.load_F.device)


            custom_grad = (self.load_F - Q) / torch.sqrt(torch.mean((self.load_F - Q) ** 2) + 1e-10)
            self.bias -= self.alpha * custom_grad




    
        
    def forward(self, x):
        """
        我吐了呀
        Args:
            x: (batch_size, input_dim)
        Returns:
            gates: (batch_size, top_k) - 门控权重
            router: (batch_size, num_experts) - 0/1指示每个样本选择的专家
            load_F: 负载分布 (num_experts,)
        """
        clean_logits = self.w_gate(x)

        if self.training and self.noise_epsilon > 0:
            # 添加噪声
            noise_logits = self.w_noise(x)
            noise = torch.randn_like(clean_logits) * F.softplus(noise_logits) * self.noise_epsilon
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
            
        # 计算偏置router
        gates = torch.sigmoid(noisy_logits)  # (batch_size, num_experts)
        bias_gates = gates + self.bias  # (batch_size, num_experts)
        _, bias_gates_index = torch.topk(bias_gates, self.top_k, dim=-1)

        gates = torch.gather(gates, 1, bias_gates_index)  # (batch_size, top_k)

        router = torch.zeros_like(noisy_logits).long() # (batch_size, num_experts)
        router.scatter_(1, bias_gates_index, torch.ones_like(bias_gates_index, dtype=torch.long)) # (batch_size, num_experts)

        load_F = None
        
        load_F = self._router_to_F(router)  # (num_experts,)
        self.load_F = load_F.detach()
        if self.training:
            self._bias_grad_hook()  # 更新bias

        return gates, router, load_F





