r"""
GLUE component modules for single-cell omics data
PROPERLY CORRECTED VERSION - Addresses Gemini's critical issues
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS
from . import glue
from .nn import GraphConv  # Now uses concatenation by default!
from .prob import ZILN, ZIN, ZINB

# ------------------------- Network modules for GLUE ---------------------------


class GraphEncoder(glue.GraphEncoder):
    r"""
    Graph encoder with PROPER PNA
    
    CORRECTED to address Gemini's findings:
    - Uses GraphConv with concatenation (not summation)
    - Enables degree scalers by default

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    use_degree_scalers
        Whether to use PNA degree scalers (DEFAULT: True now!)
    """

    def __init__(
        self, 
        vnum: int, 
        out_features: int,
        use_degree_scalers: bool = True  # CHANGED: Default True!
    ) -> None:
        super().__init__()
        
        # Initialize vertex representations
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        # Xavier initialization (not zeros)
        torch.nn.init.xavier_normal_(self.vrepr)
        
        # CORRECTED: GraphConv now uses concatenation + degree scalers by default
        # This addresses both of Gemini's critical issues
        self.conv = GraphConv(out_features, out_features, 
                             use_degree_scalers=use_degree_scalers)
        
        # Linear layers expect out_features (GraphConv projects back to this)
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
        self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        # GraphConv returns [num_nodes, out_features] after internal projection
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        
        # Standard variational encoding
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


# ============================================================================
# Rest of the file remains unchanged
# ============================================================================


class GraphDecoder(glue.GraphDecoder):
    r"""
    Graph decoder
    """

    def forward(
        self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):
    r"""
    Abstract data encoder
    """

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
    r"""
    Vanilla data encoder
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(self, x: torch.Tensor, l: Optional[torch.Tensor]) -> torch.Tensor:
        return x


class NBDataEncoder(DataEncoder):
    r"""
    Data encoder for negative binomial data
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(glue.DataDecoder):
    r"""
    Abstract data decoder
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
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
    r"""
    Normal data decoder
    """

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
    r"""
    Zero-inflated normal data decoder
    """

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
    r"""
    Zero-inflated log-normal data decoder
    """

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
    r"""
    Negative binomial data decoder
    """

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
    r"""
    Zero-inflated negative binomial data decoder
    """

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
    r"""
    Modality discriminator
    """

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

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Classifier(torch.nn.Linear):
    r"""
    Linear label classifier
    """


class Prior(glue.Prior):
    r"""
    Prior distribution
    """

    def __init__(self, loc: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


# ------------------- Network modules for independent GLUE ---------------------


class IndDataDecoder(DataDecoder):
    r"""
    Data decoder mixin that makes decoding independent of feature latent
    """

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
    r"""
    Normal data decoder independent of feature latent
    """


class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    r"""
    Zero-inflated normal data decoder independent of feature latent
    """


class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    r"""
    Zero-inflated log-normal data decoder independent of feature latent
    """


class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    r"""
    Negative binomial data decoder independent of feature latent
    """


class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder independent of feature latent
    """
