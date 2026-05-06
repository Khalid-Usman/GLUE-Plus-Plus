r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS
from . import glue
from .nn import HetGAT
from .prob import ZILN, ZIN, ZINB

# ------------------------- Network modules for GLUE ---------------------------

class GraphEncoder(glue.GraphEncoder):
    r"""
    Graph encoder with HetGAT for heterogeneous multi-omics graphs.
    """

    def __init__(
        self,
        vnum: int,
        out_features: int,
        num_heads: int = 5,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        # Vertex representation embedding
        self.vrepr = torch.nn.Parameter(torch.empty(vnum, out_features))
        
        # CRITICAL FIX: Increased gain from 0.01 to 1.414 (standard for LeakyReLU)
        # This ensures the graph signal is strong enough at initialization
        torch.nn.init.xavier_normal_(self.vrepr, gain=1.414)

        # HetGAT Convolution
        self.conv = HetGAT(
            in_features=out_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        )

        # Projection head
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(out_features, out_features),
            torch.nn.ELU(),
            torch.nn.Linear(out_features, out_features)
        )

        # Variational parameters
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
        self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        # 1. Graph Attention Processing
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        
        # 2. Projection
        ptr = self.proj(ptr)
        
        # 3. Latent Distribution
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(glue.GraphDecoder):
    def forward(
        self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        h_depth: int = 2,
        h_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError 

    @abstractmethod
    def normalize(self, x: torch.Tensor, l: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, xrep: torch.Tensor, lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
            
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(self, x: torch.Tensor, l: Optional[torch.Tensor]) -> torch.Tensor:
        return x


class NBDataEncoder(DataEncoder):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        h_depth: int = 2,
        h_dim: int = 256,
        dropout: float = 0.2,
        total_count: float = 1e4,
    ) -> None:
        super().__init__(in_features, out_features, h_depth, h_dim, dropout)
        self.total_count = total_count

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        if (x < 0).any():
            # Soft warning or handling for negative values if scaling was applied incorrectly
            pass 
        return x.sum(dim=1, keepdim=True)

    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid log of negative numbers if input is noisy
        return (x * (self.total_count / (l + 1e-6))).log1p()


class DataDecoder(glue.DataDecoder):
    def __init__(
        self, out_features: int, n_batches: int = 1
    ) -> None: 
        super().__init__()

    @abstractmethod
    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        l: Optional[torch.Tensor],
    ) -> D.Normal:
        raise NotImplementedError


class NormalDataDecoder(DataDecoder):
    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        l: Optional[torch.Tensor],
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):
    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        l: Optional[torch.Tensor],
    ) -> ZIN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):
    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        l: Optional[torch.Tensor],
    ) -> ZILN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):
    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(log_theta.exp(), logits=(mu + EPS).log() - log_theta)


class ZINBDataDecoder(NBDataDecoder):
    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        l: Optional[torch.Tensor],
    ) -> ZINB:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta,
        )


class Discriminator(torch.nn.Sequential, glue.Discriminator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_batches: int = 0,
        h_depth: int = 2,
        h_dim: Optional[int] = 256,
        dropout: float = 0.2,
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(
        self, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor: 
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Classifier(torch.nn.Linear):
    r"""Linear label classifier"""


class Prior(glue.Prior):
    def __init__(self, loc: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


class IndDataDecoder(DataDecoder):
    def __init__( 
        self, in_features: int, out_features: int, n_batches: int = 1
    ) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.v = torch.nn.Parameter(torch.zeros(out_features, in_features))

    def forward( 
        self, u: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Distribution:
        return super().forward(u, self.v, b, l)


class IndNormalDataDecoder(IndDataDecoder, NormalDataDecoder):
    pass
class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    pass
class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    pass
class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    pass
class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    pass
