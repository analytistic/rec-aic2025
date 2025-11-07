from torch import nn
import torch
from torch.nn import functional as F

class AEencoder(nn.Module):
    def __init__(self,
                 encoder_list = [1024, 512, 256],
                ):
        super(AEencoder, self).__init__()
        self.encoder_list = encoder_list
        self.encoder = nn.ModuleList()
        self.__post__init__()

    def __post__init__(self):
        for i in range(len(self.encoder_list) - 1):
            self.encoder.append(nn.Linear(self.encoder_list[i], self.encoder_list[i + 1]))
            if i < len(self.encoder_list) - 2:
                self.encoder.append(nn.GELU())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
    

class AEdecoder(nn.Module):
    def __init__(self,
                 decoder_list = [256, 512, 1024],
                ):
        super(AEdecoder, self).__init__()
        self.decoder_list = decoder_list
        self.decoder = nn.ModuleList()
        self.__post__init__()

    def __post__init__(self):
        for i in range(len(self.decoder_list) - 1):
            self.decoder.append(nn.Linear(self.decoder_list[i], self.decoder_list[i + 1]))
            if i < len(self.decoder_list) - 2:
                self.decoder.append(nn.GELU())

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
    


class RQ(nn.Module):
    """
        实现量化
    
    """
    def __init__(self,
                 codebook_list: list = [1024, 512, 256],
                 dim: int = 256,
                 beta: float = 1.0,
                 gamma: float = 0.25,
    ):
        super(RQ, self).__init__()
        self.codebooks = nn.ModuleList()
        self.beta = beta
        self.gamma = gamma
        self.__post__init__(codebook_list, dim)


    def __post__init__(self, codebook_list: list, dim: int):
        for i in range(len(codebook_list)):
            codebook = nn.Embedding(codebook_list[i], dim)
            self.codebooks.append(codebook)

    def _nearest(self, input, C):
        with torch.no_grad():
            dist = torch.cdist(input, C.weight, p=2)
            idx = torch.argmin(dist, dim=-1)
        return idx
    
    def _kmeans(self, input, K=1024, max_iter=20, atol=1e-5, rtol=0.0, generator=None):
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
        
    def _init_codebook(self, input, C, method: str):
        C.weight.data.copy_(self._kmeans(input, K=C.num_embeddings, max_iter=20, atol=1e-5, rtol=0.0))

    



    def _commitment_loss(self, input, quantized):
        z2c = F.mse_loss(input, quantized.detach())
        c2z = F.mse_loss(input.detach(), quantized)
        return self.beta * c2z + self.gamma * z2c
    
    def _EMAup(self, idx):
        """
        todo: EMA update
        """
        pass
    
        


    def forward(self, input, init_codebook: bool = False):
        residual = input.clone()
        output = torch.zeros_like(input)
        commitment_loss = []
        for i in range(len(self.codebooks)):
            C = self.codebooks[i]
            if init_codebook:
                self._init_codebook(input, C, method='kmeans')
            idx = self._nearest(residual, C)
            quantized = C(idx)
            loss = self._commitment_loss(residual, quantized)
            commitment_loss.append(loss)
            output += quantized
            residual = residual - quantized

        return output, commitment_loss
    
    def quantize(self, input):
        residual = input.clone()
        sid = []
        with torch.no_grad():
            for i in range(len(self.codebooks)):
                C = self.codebooks[i]
                idx = self._nearest(residual, C)
                sid.append(idx)
                quantized = C(idx)
                residual = residual - quantized

        return torch.stack(sid, dim=-1)














class RQVae(nn.Module):
    def __init__(self,
                 cfg,
    ):
        super(RQVae, self).__init__()
        self.cfg = cfg
        self.encoder = AEencoder(encoder_list=cfg.model.encoder_list)
        self.decoder = AEdecoder(decoder_list=cfg.model.decoder_list)
        self.rq = RQ(codebook_list=cfg.model.codebook_list, dim=cfg.model.encoder_list[-1])

    def _recon_loss(self, input, output):
        recon_loss = F.mse_loss(output, input)
        return recon_loss

    def forward(self, input):
        z = self.encoder(input)
        zq, commitment_loss = self.rq(z)
        output = self.decoder(z + zq.detach() - z.detach())
        recon_loss = self._recon_loss(input, output)
        return recon_loss, commitment_loss
    
    def quantize(self, input):
        z = self.encoder(input)
        sid = self.rq.quantize(z)
        return sid






