r"""
Neural network modules, datasets & data loaders, and other utilities
"""

import functools
import os
from math import sqrt

import numpy as np
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

from ..utils import config, logged


# ------------------------- Neural network modules -----------------------------

class HetGATLayer(nn.Module):
    r"""
    Single layer of Heterogeneous Graph Attention (Vectorized & Sign-Aware)
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 5,
                 dropout: float = 0.2, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.negative_slope = negative_slope
        
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"

        # Linear projection
        self.linear = nn.Linear(in_features, num_heads * self.head_dim, bias=False)
        
        # Attention mechanisms (Source and Target)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        
        # Fix #5: Sign-aware attention
        self.sign_weight = nn.Parameter(torch.Tensor(1, num_heads, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        
        # Fix #13: GPU-safe epsilon
        self.eps = 1e-6  # Safe for float32 on GPU
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.constant_(self.sign_weight, 0.5)  # Fix #5: Initialize sign weight

    def forward(self, x: torch.Tensor, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: Node features [N, in_features]
            eidx: Edge indices [2, E]
            enorm: Edge weights [E] (NOW USED - Fix #6)
            esgn: Edge signs [E] (+1 for activator, -1 for repressor)
        """
        N = x.size(0)
        sidx, tidx = eidx
        
        # 1. Linear Projection & Reshape -> [N, Heads, Dim]
        feat = self.linear(x).view(N, self.num_heads, self.head_dim)
        
        # 2. Attention Scores (Vectorized)
        # alpha = LeakyReLU(a_src * h_src + a_dst * h_dst)
        alpha_src = (feat * self.att_src).sum(dim=-1)
        alpha_dst = (feat * self.att_dst).sum(dim=-1)
        
        scores = alpha_src[sidx] + alpha_dst[tidx]
        scores = self.leaky_relu(scores)
        
        # 3. Stable Softmax (manually implemented for sparse graph)
        # Subtract max for numerical stability
        # Fix #14: Isolated node handling
        scores_max = torch.full((N, self.num_heads), float('-inf'), device=x.device)
        scores_max.scatter_reduce_(0, tidx.unsqueeze(-1).expand_as(scores), scores, reduce='amax', include_self=False)
        scores = scores - scores_max[tidx]
        
        exp_scores = scores.exp()
        denom = torch.zeros(N, self.num_heads, device=x.device)
        denom.scatter_add_(0, tidx.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        
        # Add epsilon to prevent division by zero for isolated nodes
        attn_weights = exp_scores / (denom[tidx] + self.eps)  # Fix #13
        attn_weights = self.dropout(attn_weights)
        
        # Fix #6: Incorporate edge weights
        combined_weights = attn_weights * enorm.view(-1, 1)
        
        # Renormalize
        denom_final = torch.zeros(N, self.num_heads, device=x.device)
        denom_final.scatter_add_(0, tidx.unsqueeze(-1).expand_as(combined_weights), combined_weights)
        final_weights = combined_weights / (denom_final[tidx] + self.eps)
        final_weights = self.dropout(final_weights)
        
        # 4. Sign-Aware Message Passing
        # Message = Attention * (Feature_src * Edge_Sign)
        # This preserves the regulatory logic: Repressors flip the signal.
        msg = feat[sidx] * esgn.view(-1, 1, 1) 
        weighted_msg = msg * attn_weights.unsqueeze(-1)
        
        # 5. Aggregation
        out = torch.zeros_like(feat)
        out.scatter_add_(0, tidx.view(-1, 1, 1).expand_as(weighted_msg), weighted_msg)
        
        return out.reshape(N, self.num_heads * self.head_dim)


class HetGAT(nn.Module):
    r"""
    Multi-layer Heterogeneous Graph Attention Network
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 5,
                 dropout: float = 0.2, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Layer 1
        self.layers.append(HetGATLayer(
            in_features=in_features, 
            out_features=out_features, 
            num_heads=num_heads, 
            dropout=dropout
        ))
        
        # Layer 2 (Output)
        # Input dim is out_features because Layer 1 outputs concatenated heads
        self.layers.append(HetGATLayer(
            in_features=out_features, 
            out_features=out_features, 
            num_heads=num_heads, 
            dropout=dropout
        ))
        
        # Residual projection if dims don't match
        self.res_proj = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor) -> torch.Tensor:
        r"""
        Note: `enorm` is accepted for API compatibility but intentionally unused 
        to allow the attention mechanism to learn topological importance.
        """
        # Residual connection
        res = self.res_proj(x)
        
        # Layer 1
        h = self.layers[0](x, eidx, enorm, esgn)  # Fix #6: Pass enorm
        h = F.elu(h)
        h = self.dropout(h)
        
        # Layer 2
        h = self.layers[1](h, eidx, enorm, esgn)  # Fix #6: Pass enorm
        
        # Fix #17: Add residual before final dropout
        h = h + res
        h = self.dropout(h)
        return h


class GraphConv(torch.nn.Module):
    r"""
    Graph convolution (propagation only) - kept for backward compatibility
    """
    def forward(self, input: torch.Tensor, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor) -> torch.Tensor:
        sidx, tidx = eidx
        message = input[sidx] * (esgn * enorm).unsqueeze(1)
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)
        res.scatter_add_(0, tidx, message)
        return res


# ---------------------------- Utility functions -------------------------------

def M(C, u, v, epsilon):
    """
    Modified cost for logarithmic updates in Sinkhorn distance
    """
    return (u.unsqueeze(-1) + v.unsqueeze(-2) - C) / epsilon

def sinkhorn_distance(x, y, epsilon=0.1, max_iter=20, reduction='mean'):
    r"""
    Compute Sinkhorn distance (approximation of Wasserstein distance).
    
    This function is required by glue.py for Optimal Transport loss calculation.
    """
    # The pseudo-metric for Sinkhorn is usually Squared Euclidean distance
    # C = ||x - y||^2
    C = torch.cdist(x, y, p=2) ** 2
    
    # Marginal distributions (uniform)
    # Fix #11: Handle different batch sizes
    batch_size_x = x.size(0)
    batch_size_y = y.size(0)
    mu = torch.empty(batch_size_x, dtype=x.dtype, requires_grad=False, device=x.device).fill_(1.0 / batch_size_x)
    nu = torch.empty(batch_size_y, dtype=x.dtype, requires_grad=False, device=x.device).fill_(1.0 / batch_size_y)

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    
    # Sinkhorn iterations
    # Fix #15: Tighter convergence threshold
    thresh = 1e-4
    for i in range(max_iter):
        u1 = u
        u = epsilon * (torch.log(mu + 1e-8) - torch.logsumexp(M(C, u, v, epsilon), dim=-1)) + u
        v = epsilon * (torch.log(nu + 1e-8) - torch.logsumexp(M(C, u, v, epsilon).transpose(-2, -1), dim=-1)) + v
        err = (u - u1).abs().sum()
        if err < thresh:
            break
            
    # Transport plan pi = exp(M(C, u, v, epsilon))
    pi = torch.exp(M(C, u, v, epsilon))
    cost = torch.sum(pi * C)
    return cost


def freeze_running_stats(m: torch.nn.Module) -> None:
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    used_device = -1
    if not config.CPU_ONLY:
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                return torch.device("cuda")
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free
                for i in range(pynvml.nvmlDeviceGetCount())
            ])
            if free_mems.size:
                for item in config.MASKED_GPUS:
                    free_mems[item] = -1
                best_devices = np.where(free_mems == free_mems.max())[0]
                used_device = np.random.choice(best_devices, 1)[0]
                if free_mems[used_device] < 0:
                    used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")
