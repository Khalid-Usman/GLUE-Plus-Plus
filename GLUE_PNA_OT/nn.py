r"""
Neural network modules, datasets & data loaders, and other utilities
CORRECTED VERSION with proper PNA implementation
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
    Graph convolution with Principal Neighbourhood Aggregation (PNA).
    
    CORRECTED VERSION: Now includes learnable transformations for each aggregator.
    This follows the original PNA paper more closely and allows the model to
    learn how to weight different aggregation functions.
    
    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    use_degree_scalers
        Whether to use degree-based scalers (following PNA paper)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        use_degree_scalers: bool = False
    ) -> None:
        super().__init__()
        
        # CRITICAL FIX: Add learnable transformations for each aggregator
        # This is what was missing in the original implementation
        self.self_lin = torch.nn.Linear(in_features, out_features)
        self.mean_lin = torch.nn.Linear(in_features, out_features)
        self.max_lin = torch.nn.Linear(in_features, out_features)
        self.min_lin = torch.nn.Linear(in_features, out_features)
        self.std_lin = torch.nn.Linear(in_features, out_features)
        
        # Add batch normalization and non-linearity
        self.bn = torch.nn.BatchNorm1d(out_features)
        
        # Optional: degree scalers as per PNA paper
        self.use_degree_scalers = use_degree_scalers
        if use_degree_scalers:
            # Each scaler multiplies the aggregated result
            # We'll apply 4 scalers: identity, amplification, attenuation, log
            self.scaler_lin = torch.nn.Linear(out_features * 4, out_features)

    def forward(
        self,
        input: torch.Tensor,
        eidx: torch.Tensor,
        enorm: torch.Tensor,
        esgn: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward propagation with PNA.

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
        sidx, tidx = eidx  # source index and target index
        num_nodes = input.size(0)
        
        # Calculate the message based on source nodes
        message = input[sidx] * (esgn * enorm).unsqueeze(1)

        # --- PNA Aggregators ---
        
        # 1. Mean aggregator
        agg_sum = torch.zeros_like(input)
        agg_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message)
        
        # Calculate degree for mean normalization
        degree = torch.zeros(num_nodes, 1, device=input.device)
        degree.scatter_add_(0, tidx.unsqueeze(1), torch.ones_like(tidx.unsqueeze(1), dtype=input.dtype))
        degree = degree.clamp(min=1.0)  # Avoid division by zero
        
        agg_mean = agg_sum / degree

        # 2. Max and Min aggregators
        initial_max = torch.full_like(input, -float('inf'))
        initial_min = torch.full_like(input, float('inf'))

        agg_max = initial_max.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), message, reduce="amax", include_self=False)
        agg_min = initial_min.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), message, reduce="amin", include_self=False)

        # Replace +/- inf with zeros for isolated nodes
        agg_max = torch.where(torch.isneginf(agg_max), torch.zeros_like(agg_max), agg_max)
        agg_min = torch.where(torch.isinf(agg_min), torch.zeros_like(agg_min), agg_min)

        # 3. Standard Deviation aggregator
        mean_sq = agg_mean.pow(2)
        sq_sum = torch.zeros_like(input)
        sq_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message.pow(2))
        sq_mean = sq_sum / degree
        
        agg_std = torch.sqrt(torch.relu(sq_mean - mean_sq) + EPS)

        # --- CRITICAL FIX: Apply learnable transformations and SUM (not concatenate) ---
        # This maintains output dimension = input dimension like standard GNN layers
        res = (
            self.self_lin(input) +      # Self-connection
            self.mean_lin(agg_mean) +   # Mean aggregation
            self.max_lin(agg_max) +     # Max aggregation
            self.min_lin(agg_min) +     # Min aggregation
            self.std_lin(agg_std)       # Std aggregation
        )
        
        # Optional: Apply degree scalers (from PNA paper)
        if self.use_degree_scalers:
            degree_features = degree.squeeze(-1)  # [num_nodes]
            log_degree = torch.log(degree_features + 1)
            
            # Four scalers: identity, amplification, attenuation, log
            scalers = torch.stack([
                torch.ones_like(degree_features),      # identity
                degree_features,                        # amplification
                1.0 / (degree_features + 1),           # attenuation
                log_degree                              # logarithmic
            ], dim=1)  # [num_nodes, 4]
            
            # Apply scalers by broadcasting
            scaled = res.unsqueeze(1) * scalers.unsqueeze(-1)  # [num_nodes, 4, out_features]
            scaled = scaled.view(num_nodes, -1)  # [num_nodes, 4*out_features]
            res = self.scaler_lin(scaled)
        
        # Apply batch normalization and non-linearity
        res = self.bn(res)
        res = F.leaky_relu(res, negative_slope=0.2)
        
        return res


class GraphConvConcatenate(torch.nn.Module):
    r"""
    Alternative PNA implementation using concatenation instead of summation.
    This is closer to your original implementation but with learnable transformations.
    
    Use this if you want to maintain the concatenation approach but fix the
    missing learnable parameters issue.
    
    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """
    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        
        # Since we concatenate 5 aggregators, we need a projection layer
        # that maps 5*in_features -> out_features
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(5 * in_features, out_features),
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
        Forward propagation with concatenation-based PNA.
        """
        sidx, tidx = eidx
        num_nodes = input.size(0)
        
        message = input[sidx] * (esgn * enorm).unsqueeze(1)

        # Aggregators (same as before)
        agg_sum = torch.zeros_like(input)
        agg_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message)
        
        degree = torch.zeros(num_nodes, 1, device=input.device)
        degree.scatter_add_(0, tidx.unsqueeze(1), torch.ones_like(tidx.unsqueeze(1), dtype=input.dtype))
        degree = degree.clamp(min=1.0)
        
        agg_mean = agg_sum / degree

        initial_max = torch.full_like(input, -float('inf'))
        initial_min = torch.full_like(input, float('inf'))

        agg_max = initial_max.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), message, reduce="amax", include_self=False)
        agg_min = initial_min.scatter_reduce(0, tidx.unsqueeze(1).expand_as(message), message, reduce="amin", include_self=False)

        agg_max = torch.where(torch.isneginf(agg_max), torch.zeros_like(agg_max), agg_max)
        agg_min = torch.where(torch.isinf(agg_min), torch.zeros_like(agg_min), agg_min)

        mean_sq = agg_mean.pow(2)
        sq_sum = torch.zeros_like(input)
        sq_sum.scatter_add_(0, tidx.unsqueeze(1).expand_as(message), message.pow(2))
        sq_mean = sq_sum / degree
        
        agg_std = torch.sqrt(torch.relu(sq_mean - mean_sq) + EPS)

        # Concatenate and project (FIXED with learnable transformation)
        res = torch.cat([input, agg_mean, agg_max, agg_min, agg_std], dim=1)
        res = self.projection(res)
        
        return res


class GraphAttent(torch.nn.Module):  # pragma: no cover

    r"""
    Graph attention

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality

    Note
    ----
    **EXPERIMENTAL**
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.ParameterDict(
            {
                "pos": torch.nn.Parameter(torch.Tensor(out_features, in_features)),
                "neg": torch.nn.Parameter(torch.Tensor(out_features, in_features)),
            }
        )
        self.head = torch.nn.ParameterDict(
            {
                "pos": torch.nn.Parameter(torch.zeros(out_features * 2)),
                "neg": torch.nn.Parameter(torch.zeros(out_features * 2)),
            }
        )
        torch.nn.init.kaiming_uniform_(
            self.weight["pos"], sqrt(5)
        )
        torch.nn.init.kaiming_uniform_(
            self.weight["neg"], sqrt(5)
        )

    def forward(
        self,
        input: torch.Tensor,
        eidx: torch.Tensor,
        ewt: torch.Tensor,
        esgn: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph attention result (:math:`n_{vertices} \times n_{features}`)
        """
        res_dict = {}
        for sgn in ("pos", "neg"):
            mask = esgn == 1 if sgn == "pos" else esgn == -1
            sidx, tidx = eidx[:, mask]
            ptr = input @ self.weight[sgn].T
            alpha = torch.cat([ptr[sidx], ptr[tidx]], dim=1) @ self.head[sgn]
            alpha = F.leaky_relu(alpha, negative_slope=0.2).exp() * ewt[mask]
            normalizer = torch.zeros(ptr.shape[0], device=ptr.device)
            normalizer.scatter_add_(0, tidx, alpha)
            alpha = (
                alpha / normalizer[tidx]
            )
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

    Parameters
    ----------
    m
        Network module
    """
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    based on GPU availability and memory usage

    Returns
    -------
    device
        Computation device
    """
    used_device = -1
    if not config.CPU_ONLY:
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                return torch.device("cuda")
            pynvml.nvmlInit()
            free_mems = np.array(
                [
                    pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(i)
                    ).free
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            )
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
