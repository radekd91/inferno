# Code adapted from [Ng et al 2022] Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion: 
# https://github.com/evonneng/learning2listen/blob/main/src/vqgan/vqmodules/quantizer.py
# They adapted it from [Esser, Rombach 2021]: https://compvis.github.io/taming-transformers/

import torch
import torch.nn as nn
from .MotionPrior import MotionQuantizer
from torch.nn import functional as F
from typing import Dict
from inferno.utils.ValueScheduler import scheduler_from_dict

def kl_divergence(p, q, reduction='batchmean'):
    """
    Computes the KL divergence between two distributions.
    Inputs:
    - p : tensor of shape (batch_size, num_classes), p is expected to be the distribution of the ground truth
    - q : tensor of shape (batch_size, num_classes), q is expected to be the output of a model
    - reduction : string, either 'none', 'mean', or 'sum'
    """
    logp = torch.log(p + 1e-10)
    logq = torch.log(q + 1e-10)
    kl_divergence = p * (logp - logq) 
    if reduction == 'none':
        return kl_divergence
    elif reduction == 'mean':
        return torch.mean(kl_divergence)
    elif reduction == 'sum':
        return torch.sum(kl_divergence)
    elif reduction == 'batchmean':
        return torch.mean(torch.sum(kl_divergence, dim=1))
    return kl_divergence


class GumbelVectorQuantizer(MotionQuantizer):
    """
    Discretization bottleneck part of the dVAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.codebook_size = cfg.codebook_size
        self.vector_dim = cfg.vector_dim
        self.tau = scheduler_from_dict(cfg.tau)
        self.embedding = nn.Embedding(self.codebook_size, self.vector_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, batch, input_key="encoded_features", output_key="quantized_features", tau=None, step=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,T,D)
            2. flatten input to (B*T,D)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        #print('zshape', z.shape)
        z = batch[input_key]
        # z = z.permute(0, 2, 1).contiguous()
        B, T = z.shape[:2]
        z_flattened = z.view(B*T, -1)

        # compute the soft assignments using gumbel softmax
        tau = tau or self.tau(step=step)
        soft_assignments = F.gumbel_softmax(z_flattened, tau=tau, hard=False)
        
        # get the linear combination of the codebook vectors
        z_q = soft_assignments @ self.embedding.weight
        z_q = z_q.view(B, T, -1)
        batch[output_key] = z_q

        # # compute loss for embedding 
        # # TODO: do we want this here? probably not necessary for the DVAE
        # codebook_alignment = torch.mean((z_q.detach()-z)**2)
        # codebook_commitment = torch.mean((z_q - z.detach()) ** 2)

        # loss = self.beta * codebook_alignment + \
        #           codebook_commitment
        # #loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        # #    torch.mean((z_q - z.detach()) ** 2)

        # # preserve gradients
        # z_q = z + (z_q - z).detach()
        uniform_dist = torch.ones_like(soft_assignments) / self.codebook_size
        kl_div = kl_divergence(uniform_dist, soft_assignments)

        min_encoding_indices = torch.argmax(soft_assignments, dim=1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # # soft perplexity ... what am I really doing here?! :-D 
        # soft_perplexity = torch.exp(-torch.sum(z_q * torch.log(z_q + 1e-10), dim=1))

        batch[output_key] = z_q
        batch["kl_divergence"] = kl_div
        batch["perplexity"] = perplexity
        # batch["soft_perplexity"] = soft_perplexity
        batch["soft_assignments"] = soft_assignments
        batch["min_encoding_indices"] = min_encoding_indices 
        batch["min_encodings"] = min_encodings
        batch["gumble_tau"] = tau
        # batch["codebook_alignment"] = codebook_alignment
        # batch["codebook_commitment"] = codebook_commitment
        return batch


    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.vector_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        #print(min_encodings.shape, self.embedding.weight.shape)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
