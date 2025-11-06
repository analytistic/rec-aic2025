
import torch.nn as nn
import torch

class RQKmeans(nn.Module):
    def __init__(self,
                 cfg,
                ):
        super(RQKmeans, self).__init__()
        self.cfg = cfg
        self.num_book = cfg.model.num_book
        self.num_cluster = cfg.model.num_cluster
        self.seed = cfg.seed
        self.codebooks = [getattr(self, f'codebook_{i}') for i in range(self.num_book)]
        self.device = cfg.device
        


    def kmeans(self, input, K=1024, max_iter=20, atol=1e-5, rtol=0.0, generator=None):
        N, D = input.shape
        C = input[torch.randperm(N, generator=generator)[:K], :].clone()
        for _ in range(max_iter):
            dist = torch.cdist(input, C, p=2) # N, K
            index = torch.argmin(dist, dim=-1)
            C_new = torch.zeros_like(C)
            counts = torch.bincount(index, minlength=K).clamp(min=1).unsqueeze(-1).to(C.dtype)
            C_new.index_add_(0, index, input)
            C_new = C_new/counts

            if torch.allclose(C, C_new, atol=atol, rtol=rtol):
                C = C_new
                break
            C = C_new
        return C
    
    @torch.no_grad()
    @staticmethod
    def nearest(input, C):
        dist = torch.cdist(input, C, p=2)  # N, K
        index = torch.argmin(dist, dim=-1)
        return index
    
    @torch.no_grad()
    def fit(self, input):
        assert input.dim() == 2, "Input tensor must be 2-dimensional"

        residual = input.clone()
    
        codebooks = []
        for b in range(self.num_book):
            C = self.kmeans(residual, K=self.num_cluster[b], max_iter=20, atol=1e-5, rtol=0.0,
                            generator=torch.Generator().manual_seed(self.seed + b))
            codebooks.append(C.to(self.device))
            residual = residual - C[self.nearest(residual, C)]

        self.codebooks = codebooks
        
        for i in range(self.num_book):
            self.register_buffer(f'codebook_{i}', codebooks[i])






    @torch.no_grad()
    def forward(self, input):
        assert input.dim() == 2, "Input tensor must be 2-dimensional"
        assert self.codebooks, "Codebooks must be initialized"
        output = []
        for i in range(self.num_book):
            idx = self.nearest(input, self.codebooks[i])
            output.append(idx)

        output = torch.stack(output, dim=-1)
        return output

        
        

    @torch.no_grad()
    def decoder(self, sid):
        assert sid.dim() == 2, "Input tensor must be 2-dimensional"
        output = 0
        for i in range(self.num_book):
            codebook = getattr(self, f'codebook_{i}')
            output = output + codebook[sid[:, i]]
        return output
