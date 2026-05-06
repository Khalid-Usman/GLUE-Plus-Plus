r"""
Neural network modules - PROPERLY CORRECTED VERSION
Fixes both Gemini's critical issues:
1. Uses CONCATENATION (not summation)
2. Enables DEGREE SCALERS by default
"""

import functools
import os
from math import sqrt

import numpy as np
import pynvml
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

from ..utils import config, logged
from ..num import EPS

# ------------------------- Neural network modules -----------------------------


class GraphConv(torch.nn.Module):
    r"""
    PROPER PNA Implementation with Concatenation and Degree Scalers
    
    This fixes Gemini's identified issues:
    1. Uses concatenation (not summation) following original PNA paper
    2. Enables degree scalers by default
    
    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality  
    use_degree_scalers
        Whether to use PNA degree scalers (DEFAULT: True)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        use_degree_scalers: bool = True  # CHANGED: Default True!
    ) -> None:
        super().__init__()
        
        # Separate transformations for each aggregator
        self.self_lin = torch.nn.Linear(in_features, out_features)
        self.mean_lin = torch.nn.Linear(in_features, out_features)
        self.max_lin = torch.nn.Linear(in_features, out_features)
        self.min_lin = torch.nn.Linear(in_features, out_features)
        self.std_lin = torch.nn.Linear(in_features, out_features)
        
        # Calculate concatenated dimension
        # 5 aggregators × out_features each
        concat_dim = 5 * out_features
        
        # Degree scalers (ENABLED by default)
        self.use_degree_scalers = use_degree_scalers
        if use_degree_scalers:
            # 4 scalers: identity, amplification, attenuation, log
            # Each scaler creates a copy of the concatenated features
            concat_dim *= 4  # Now 20*out_features

        # Final projection from concatenated features to output
        # FIX #1: This addresses Gemini's "summation bottleneck" issue
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(concat_dim, out_features),
            torch.nn.BatchNorm1d(out_features),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(
        self,
        input: torch.Tensor,
        eidx: torch.Tensor,
        enorm: torch.Tensor,
        esgn: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward propagation with PROPER PNA (concatenation + scalers).

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            PNA result (:math:`n_{vertices} \times n_{out_features}`)
        """
        sidx, tidx = eidx
        num_nodes = input.size(0)
        
        message = input[sidx] * (esgn * enorm).unsqueeze(1)

        # === PNA Aggregators ===
        
        # 1. Mean aggregator
        agg_sum = torch.zeros_like(input)
        agg_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message)
        
        degree = torch.zeros(num_nodes, 1, device=input.device)
        degree.scatter_add_(0, tidx.unsqueeze(1), 
                           torch.ones_like(tidx.unsqueeze(1), dtype=input.dtype))
        degree = degree.clamp(min=1.0)
        
        agg_mean = agg_sum / degree

        # 2. Max and Min aggregators
        initial_max = torch.full_like(input, -float('inf'))
        initial_min = torch.full_like(input, float('inf'))

        agg_max = initial_max.scatter_reduce(
            0, tidx.unsqueeze(1).expand_as(message), message, 
            reduce="amax", include_self=False
        )
        agg_min = initial_min.scatter_reduce(
            0, tidx.unsqueeze(1).expand_as(message), message, 
            reduce="amin", include_self=False
        )

        agg_max = torch.where(torch.isneginf(agg_max), torch.zeros_like(agg_max), agg_max)
        agg_min = torch.where(torch.isinf(agg_min), torch.zeros_like(agg_min), agg_min)

        # 3. Standard Deviation aggregator
        mean_sq = agg_mean.pow(2)
        sq_sum = torch.zeros_like(input)
        sq_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message.pow(2))
        sq_mean = sq_sum / degree
        
        agg_std = torch.sqrt(torch.relu(sq_mean - mean_sq) + EPS)

        # === Transform Each Aggregator ===
        h_self = self.self_lin(input)
        h_mean = self.mean_lin(agg_mean)
        h_max = self.max_lin(agg_max)
        h_min = self.min_lin(agg_min)
        h_std = self.std_lin(agg_std)
        
        # === FIX #1: CONCATENATE (not sum!) ===
        # This is the key fix for Gemini's Issue #1
        h_cat = torch.cat([h_self, h_mean, h_max, h_min, h_std], dim=1)
        # Shape: [num_nodes, 5*out_features]
        
        # === FIX #2: Apply Degree Scalers ===
        # This is the key fix for Gemini's Issue #2
        if self.use_degree_scalers:
            degree_feat = degree.squeeze(-1)  # [num_nodes]
            log_degree = torch.log(degree_feat + 1)
            
            # Four degree scalers from PNA paper
            scalers = torch.stack([
                torch.ones_like(degree_feat),       # Identity (δ=1)
                degree_feat,                         # Amplification (δ=d)
                1.0 / (degree_feat + 1),            # Attenuation (δ=1/d)
                log_degree                           # Logarithmic (δ=log(d))
            ], dim=1)  # [num_nodes, 4]
            
            # Apply each scaler to the concatenated features
            scaled_list = []
            for i in range(4):
                # Broadcast scaler across all features
                scaled = h_cat * scalers[:, i].unsqueeze(1)
                scaled_list.append(scaled)
            
            # Concatenate all scaled versions
            h_cat = torch.cat(scaled_list, dim=1)
            # Shape: [num_nodes, 20*out_features]
        
        # === Final Projection ===
        res = self.projection(h_cat)
        # Shape: [num_nodes, out_features]
        
        return res


# Keep old version for backward compatibility
class GraphConvSummation(torch.nn.Module):
    r"""
    OLD VERSION using summation (kept for backward compatibility)
    NOT RECOMMENDED - use GraphConv instead
    """
    
    def __init__(self, in_features: int, out_features: int, use_degree_scalers: bool = False):
        super().__init__()
        
        self.self_lin = torch.nn.Linear(in_features, out_features)
        self.mean_lin = torch.nn.Linear(in_features, out_features)
        self.max_lin = torch.nn.Linear(in_features, out_features)
        self.min_lin = torch.nn.Linear(in_features, out_features)
        self.std_lin = torch.nn.Linear(in_features, out_features)
        
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.use_degree_scalers = use_degree_scalers
        
        if use_degree_scalers:
            self.scaler_lin = torch.nn.Linear(out_features * 4, out_features)

    def forward(self, input, eidx, enorm, esgn):
        sidx, tidx = eidx
        num_nodes = input.size(0)
        
        message = input[sidx] * (esgn * enorm).unsqueeze(1)

        # Aggregations (same as GraphConv)
        agg_sum = torch.zeros_like(input)
        agg_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message)
        
        degree = torch.zeros(num_nodes, 1, device=input.device)
        degree.scatter_add_(0, tidx.unsqueeze(1), 
                           torch.ones_like(tidx.unsqueeze(1), dtype=input.dtype))
        degree = degree.clamp(min=1.0)
        agg_mean = agg_sum / degree

        initial_max = torch.full_like(input, -float('inf'))
        initial_min = torch.full_like(input, float('inf'))
        agg_max = initial_max.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), 
                                             message, reduce="amax", include_self=False)
        agg_min = initial_min.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), 
                                             message, reduce="amin", include_self=False)
        agg_max = torch.where(torch.isneginf(agg_max), torch.zeros_like(agg_max), agg_max)
        agg_min = torch.where(torch.isinf(agg_min), torch.zeros_like(agg_min), agg_min)

        mean_sq = agg_mean.pow(2)
        sq_sum = torch.zeros_like(input)
        sq_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message.pow(2))
        sq_mean = sq_sum / degree
        agg_std = torch.sqrt(torch.relu(sq_mean - mean_sq) + EPS)

        # OLD: Summation (causes bottleneck)
        res = (
            self.self_lin(input) + 
            self.mean_lin(agg_mean) + 
            self.max_lin(agg_max) + 
            self.min_lin(agg_min) + 
            self.std_lin(agg_std)
        )
        
        if self.use_degree_scalers:
            degree_features = degree.squeeze(-1)
            log_degree = torch.log(degree_features + 1)
            scalers = torch.stack([
                torch.ones_like(degree_features),
                degree_features,
                1.0 / (degree_features + 1),
                log_degree
            ], dim=1)
            scaled = res.unsqueeze(1) * scalers.unsqueeze(-1)
            scaled = scaled.view(num_nodes, -1)
            res = self.scaler_lin(scaled)
        
        res = self.bn(res)
        res = F.leaky_relu(res, negative_slope=0.2)
        
        return res


class GraphAttent(torch.nn.Module):  # pragma: no cover
    r"""
    Graph attention (kept for compatibility)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.ParameterDict({
            "pos": torch.nn.Parameter(torch.Tensor(out_features, in_features)),
            "neg": torch.nn.Parameter(torch.Tensor(out_features, in_features)),
        })
        self.head = torch.nn.ParameterDict({
            "pos": torch.nn.Parameter(torch.zeros(out_features * 2)),
            "neg": torch.nn.Parameter(torch.zeros(out_features * 2)),
        })
        torch.nn.init.kaiming_uniform_(self.weight["pos"], sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight["neg"], sqrt(5))

    def forward(self, input, eidx, ewt, esgn):
        res_dict = {}
        for sgn in ("pos", "neg"):
            mask = esgn == 1 if sgn == "pos" else esgn == -1
            sidx, tidx = eidx[:, mask]
            ptr = input @ self.weight[sgn].T
            alpha = torch.cat([ptr[sidx], ptr[tidx]], dim=1) @ self.head[sgn]
            alpha = F.leaky_relu(alpha, negative_slope=0.2).exp() * ewt[mask]
            normalizer = torch.zeros(ptr.shape[0], device=ptr.device)
            normalizer.scatter_add_(0, tidx, alpha)
            alpha = alpha / normalizer[tidx]
            message = ptr[sidx] * alpha.unsqueeze(1)
            res = torch.zeros_like(ptr)
            tidx = tidx.unsqueeze(1).expand_as(message)
            res.scatter_add_(0, tidx, message)
            res_dict[sgn] = res
        return res_dict["pos"] + res_dict["neg"]


# ---------------------------- Utility functions -------------------------------


def freeze_running_stats(m: torch.nn.Module) -> None:
    r"""
    Selectively stops normalization layers from updating running stats
    """
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    """
    used_device = -1
    if not config.CPU_ONLY:
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                return torch.device("cuda")
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free
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
