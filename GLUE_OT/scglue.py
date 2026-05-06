r"""
Graph-linked unified embedding (GLUE) for single-cell multi-omics
data integration
"""

import copy
import os
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple, Union

import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData

try:
    import ot
except ImportError:
    ot = None

from ..graph import check_graph
from ..num import normalize_edges
from ..utils import AUTO, config, get_chained_attr, logged
from . import sc
from .base import Model
from .data import AnnDataset, ArrayDataset, DataLoader, GraphDataset
from .glue import GLUE, GLUETrainer
from .nn import freeze_running_stats

# --------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)


# ---------------------------- Network definition ------------------------------


class SCGLUE(GLUE):

    r"""
    GLUE network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator (optional, only for adversarial alignment)
    prior
        Latent prior
    u2c
        Data classifier (optional)
    """

    def __init__(
        self,
        g2v: sc.GraphEncoder,
        v2g: sc.GraphDecoder,
        x2u: Mapping[str, sc.DataEncoder],
        u2x: Mapping[str, sc.DataDecoder],
        idx: Mapping[str, torch.Tensor],
        du: Optional[sc.Discriminator],
        prior: sc.Prior,
        u2c: Optional[sc.Classifier] = None,
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)
        self.u2c = u2c.to(self.device) if u2c else None


class IndSCGLUE(SCGLUE):

    r"""
    GLUE network where cell and feature in different modalities are independent

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator (optional, only for adversarial alignment)
    prior
        Latent prior
    u2c
        Data classifier (optional)
    """

    def __init__(
        self,
        g2v: sc.GraphEncoder,
        v2g: sc.GraphDecoder,
        x2u: Mapping[str, sc.DataEncoder],
        u2x: Mapping[str, sc.IndDataDecoder],
        idx: Mapping[str, torch.Tensor],
        du: Optional[sc.Discriminator],
        prior: sc.Prior,
        u2c: Optional[sc.Classifier] = None,
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior, u2c)


# ---------------------------- Trainer definition ------------------------------

DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor,  # esgn (edge sign)
]  # Specifies the data format of input to SCGLUETrainer.compute_losses


@logged
class SCGLUETrainer(GLUETrainer):

    r"""
    Trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    alignment_method
        Alignment method, one of ``{"adversarial", "ot"}``
    lam_ot
        Optimal transport alignment weight
    ot_reg
        Optimal transport regularization
    ot_dtype
        Data type for OT computation, one of ``{"float32", "float64"}``
    gradient_clip_val
        Value for gradient clipping
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration

    def __init__(
        self,
        net: SCGLUE,
        lam_data: float = None,
        lam_kl: float = None,
        lam_graph: float = None,
        lam_align: float = None,
        lam_sup: float = None,
        alignment_method: str = "adversarial",
        lam_ot: float = 0.1,
        ot_reg: float = 0.1,
        ot_dtype: str = 'float32',
        gradient_clip_val: Optional[float] = None,
        normalize_u: bool = None,
        modality_weight: Mapping[str, float] = None,
        optim: str = None,
        lr: float = None,
        **kwargs,
    ) -> None:
        super().__init__(
            net,
            lam_data=lam_data,
            lam_kl=lam_kl,
            lam_graph=lam_graph,
            lam_align=lam_align,
            modality_weight=modality_weight,
            optim=optim,
            lr=lr,
            alignment_method=alignment_method,  # Pass to parent
            **kwargs,
        )
        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        # alignment_method already set by parent __init__
        if self.alignment_method == "ot":
            if ot is None:
                raise ImportError("Optimal transport alignment requires `pot` library. Please install it via `pip install pot`.")
            self.required_losses.append("ot_loss")
            self.lam_ot = lam_ot
            self.ot_reg = ot_reg
            self.ot_dtype = torch.float64 if ot_dtype == 'float64' else torch.float32
        elif self.alignment_method != "adversarial":
            raise ValueError(f"Unknown alignment method: {self.alignment_method}")

        self.lam_sup = lam_sup
        self.gradient_clip_val = gradient_clip_val
        self.normalize_u = normalize_u
        self.freeze_u = False

        vae_params = [
            self.net.g2v.parameters(),
            self.net.v2g.parameters(),
            self.net.x2u.parameters(),
            self.net.u2x.parameters(),
        ]
        if net.u2c:
            self.required_losses.append("sup_loss")
            vae_params.append(self.net.u2c.parameters())

        self.vae_optim = getattr(torch.optim, optim)(
            chain(*vae_params), lr=self.lr, **kwargs
        )

        # Discriminator optimizer already created by parent if needed
        # No need to create it here

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in self.net.x2u.parameters():
            item.requires_grad_(not self._freeze_u)
        # Only freeze discriminator if it exists (adversarial alignment)
        if self.net.du is not None:
            for item in self.net.du.parameters():
                item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, (eidx, ewt, esgn) = (
            data[0:K],
            data[K : 2 * K],
            data[2 * K : 3 * K],
            data[3 * K : 4 * K],
            data[4 * K : 5 * K],
            data[5 * K + 1 :],
        )
        x = {k: x[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xrep = {k: xrep[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xbch = {k: xbch[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xlbl = {k: xlbl[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xdwt = {k: xdwt[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xflag = {
            k: torch.as_tensor(i, dtype=torch.int64, device=device).expand(
                x[k].shape[0]
            )
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn

    def compute_losses(
        self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        u_mean_dict = {k: u[k].mean for k in net.keys}
        u_cat = torch.cat(list(u_mean_dict.values()))

        dsc_loss = torch.tensor(0.0, device=net.device)
        ot_loss = torch.tensor(0.0, device=net.device)

        # Calculate Annealing factor
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) if self.align_burnin else 0
        
        # Calculate Alignment Weight (0 during pretraining/burnin)
        # Pretraining is identified by infinite burnin
        alignment_weight = 1.0
        if self.align_burnin and (np.isinf(self.align_burnin) or self.align_burnin > 1e6):
            alignment_weight = 0.0

        # --- ADVERSARIAL LOSS ---
        # Must always be computed if alignment_method is 'adversarial',
        # because train_step calls backward() on it.
        if self.alignment_method == "adversarial":
            if self.net.du is None:
                raise RuntimeError("Discriminator network required for adversarial alignment but not provided!")
            
            xbch_cat = torch.cat([xbch[k] for k in net.keys])
            xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
            xflag_cat = torch.cat([xflag[k] for k in net.keys])
            
            if anneal:
                noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0],))
                u_cat_noised = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
            else:
                u_cat_noised = u_cat
                
            dsc_loss_raw = F.cross_entropy(net.du(u_cat_noised, xbch_cat), xflag_cat, reduction="none")
            dsc_loss = (dsc_loss_raw * xdwt_cat).sum() / xdwt_cat.numel()

        # --- OT LOSS ---
        # Compute OT loss always (will be weighted by alignment_weight later)
        # This allows gradients to flow during pretraining for better warm-up
        elif self.alignment_method == "ot":
            keys = list(net.keys)
            pairs_count = 0
            cumulative_ot = torch.tensor(0.0, device=net.device)
            
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    u1 = u_mean_dict[k1]
                    u2 = u_mean_dict[k2]
                    
                    # FIX #2: Apply normalization if enabled (matches usamp normalization)
                    if self.normalize_u:
                        u1 = F.normalize(u1, dim=1)
                        u2 = F.normalize(u2, dim=1)
                    
                    # FIX #3: Check for empty batches
                    if u1.shape[0] == 0 or u2.shape[0] == 0:
                        continue
                    
                    M = torch.cdist(u1, u2, p=2).pow(2)
                    
                    # FIX #3: Add numerical safeguards
                    M = torch.clamp(M, max=1e6)  # Prevent explosion
                    if torch.isnan(M).any() or torch.isinf(M).any():
                        self.logger.warning(f"NaN/Inf in OT cost matrix for {k1}-{k2}, skipping this pair")
                        continue
                    
                    a = (torch.ones(u1.shape[0], device=net.device) / u1.shape[0]).to(self.ot_dtype)
                    b = (torch.ones(u2.shape[0], device=net.device) / u2.shape[0]).to(self.ot_dtype)
                    M_cast = M.to(self.ot_dtype)

                    # Use sinkhorn with gradients flowing through M
                    try:
                        gamma = ot.sinkhorn(a, b, M_cast, reg=self.ot_reg, method='sinkhorn_stabilized')
                        
                        # loss = sum(gamma * M)
                        current_pair_loss = torch.sum(gamma.to(M.dtype) * M)
                        cumulative_ot += current_pair_loss
                        pairs_count += 1
                    except Exception as e:
                        self.logger.warning(f"Sinkhorn failed for {k1}-{k2}: {e}")
                        continue
            
            if pairs_count > 0:
                ot_loss = cumulative_ot / pairs_count

        # --- EARLY RETURN ---
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        # --- VAE RECONSTRUCTION LOSS ---
        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos, n_neg = pos_mask.sum().item(), pos_mask.numel() - pos_mask.sum().item()
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device).scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        x_nll = {
            k: -net.u2x[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {k: D.kl_divergence(u[k], prior).sum(dim=1).mean() / x[k].shape[1] for k in net.keys}
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        vae_loss = (self.lam_data * x_elbo_sum + self.lam_graph * len(net.keys) * g_elbo + self.lam_sup * sup_loss)

        # --- GENERATOR LOSS COMBINATION ---
        if self.alignment_method == "adversarial":
            # Multiply by alignment_weight to effectively disable it during pretraining
            # (since alignment_weight is 0 during pretraining)
            gen_loss = vae_loss - (self.lam_align * alignment_weight) * dsc_loss 
        elif self.alignment_method == "ot":
            gen_loss = vae_loss + (self.lam_ot * alignment_weight) * ot_loss
        else:
            gen_loss = vae_loss

        losses = {
            "dsc_loss": dsc_loss, "ot_loss": ot_loss, "vae_loss": vae_loss,
            "gen_loss": gen_loss, "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
        }
        for k in net.keys:
            losses.update({f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]})
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def train_step(self, engine: ignite.engine.Engine, data: List[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        if self.alignment_method == "adversarial":
            # Adversarial alignment path
            if self.freeze_u:
                self.net.x2u.apply(freeze_running_stats)
                if self.net.du is not None:
                    self.net.du.apply(freeze_running_stats)
            else:
                losses = self.compute_losses(data, epoch, dsc_only=True)
                self.net.zero_grad(set_to_none=True)
                losses["dsc_loss"].backward()
                if self.dsc_optim is not None:
                    self.dsc_optim.step()
            losses = self.compute_losses(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()
        else:
            # OT alignment path - no discriminator step
            losses = self.compute_losses(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()

        if self.gradient_clip_val is not None:
            all_vae_params = chain(
                self.net.g2v.parameters(), self.net.v2g.parameters(),
                self.net.x2u.parameters(), self.net.u2x.parameters(),
                self.net.u2c.parameters() if self.net.u2c else []
            )
            torch.nn.utils.clip_grad_norm_(all_vae_params, self.gradient_clip_val)
        
        self.vae_optim.step()
        return losses

    def __repr__(self):
        optim_repr = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        if self.alignment_method == "adversarial":
            if self.dsc_optim is not None:
                dsc_optim_repr = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
            else:
                dsc_optim_repr = "None (discriminator not created)"
            align_loss_lam, align_loss_name = self.lam_align, "lam_align"
        else:
            dsc_optim_repr, align_loss_lam, align_loss_name = "N/A (OT alignment)", self.lam_ot, "lam_ot"
        
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n  {align_loss_name}: {align_loss_lam}\n"
            f"  alignment_method: {self.alignment_method}\n  gradient_clip_val: {self.gradient_clip_val}\n"
            f"  vae_optim: {optim_repr}\n  dsc_optim: {dsc_optim_repr}\n  freeze_u: {self.freeze_u}\n)"
        )


PairedDataTensors = Tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@logged
class PairedSCGLUETrainer(SCGLUETrainer):

    r"""
    Paired trainer for :class:`SCGLUE`
    """

    def __init__(
        self,
        net: SCGLUE,
        lam_data: float = None,
        lam_kl: float = None,
        lam_graph: float = None,
        lam_align: float = None,
        lam_sup: float = None,
        alignment_method: str = "adversarial",
        lam_ot: float = 0.1,
        ot_reg: float = 0.1,
        ot_dtype: str = 'float32',
        gradient_clip_val: Optional[float] = None,
        lam_joint_cross: float = None,
        lam_real_cross: float = None,
        lam_cos: float = None,
        normalize_u: bool = None,
        modality_weight: Mapping[str, float] = None,
        optim: str = None,
        lr: float = None,
        **kwargs,
    ) -> None:
        super().__init__(
            net,
            lam_data=lam_data,
            lam_kl=lam_kl,
            lam_graph=lam_graph,
            lam_align=lam_align,
            lam_sup=lam_sup,
            alignment_method=alignment_method,
            lam_ot=lam_ot,
            ot_reg=ot_reg,
            ot_dtype=ot_dtype,
            gradient_clip_val=gradient_clip_val,
            normalize_u=normalize_u,
            modality_weight=modality_weight,
            optim=optim,
            lr=lr,
            **kwargs,
        )
        required_kwargs = ("lam_joint_cross", "lam_real_cross", "lam_cos")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_joint_cross, self.lam_real_cross, self.lam_cos = lam_joint_cross, lam_real_cross, lam_cos
        self.required_losses += ["joint_cross_loss", "real_cross_loss", "cos_loss"]

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, xflag, pmsk, (eidx, ewt, esgn) = (
            data[0:K], data[K: 2 * K], data[2 * K: 3 * K], data[3 * K: 4 * K],
            data[4 * K: 5 * K], data[5 * K], data[5 * K + 1:],
        )
        x = {k: x[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xrep = {k: xrep[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xbch = {k: xbch[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xlbl = {k: xlbl[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xdwt = {k: xdwt[i].to(device, non_blocking=True) for i, k in enumerate(keys)}
        xflag = {
            k: torch.as_tensor(i, dtype=torch.int64, device=device).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        pmsk = pmsk.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, pmsk, eidx, ewt, esgn

    def compute_losses(
        self, data: PairedDataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, pmsk, eidx, ewt, esgn = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        u_mean_dict = {k: u[k].mean for k in net.keys}
        u_cat = torch.cat(list(u_mean_dict.values()))

        dsc_loss = torch.tensor(0.0, device=net.device)
        ot_loss = torch.tensor(0.0, device=net.device)

        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) if self.align_burnin else 0
        alignment_weight = 1.0
        if self.align_burnin and (np.isinf(self.align_burnin) or self.align_burnin > 1e6):
            alignment_weight = 0.0

        if self.alignment_method == "adversarial":
            if self.net.du is None:
                raise RuntimeError("Discriminator network required for adversarial alignment but not provided!")
            
            xbch_cat = torch.cat([xbch[k] for k in net.keys])
            xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
            xflag_cat = torch.cat([xflag[k] for k in net.keys])
            
            if anneal:
                noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0],))
                u_cat_noised = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
            else:
                u_cat_noised = u_cat
                
            dsc_loss_raw = F.cross_entropy(net.du(u_cat_noised, xbch_cat), xflag_cat, reduction="none")
            dsc_loss = (dsc_loss_raw * xdwt_cat).sum() / xdwt_cat.numel()

        elif self.alignment_method == "ot":
            keys = list(net.keys)
            pairs_count = 0
            cumulative_ot = torch.tensor(0.0, device=net.device)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    u1 = u_mean_dict[k1]
                    u2 = u_mean_dict[k2]
                    
                    # FIX #2: Apply normalization if enabled (matches usamp normalization)
                    if self.normalize_u:
                        u1 = F.normalize(u1, dim=1)
                        u2 = F.normalize(u2, dim=1)
                    
                    # FIX #3: Check for empty batches
                    if u1.shape[0] == 0 or u2.shape[0] == 0:
                        continue
                        
                    M = torch.cdist(u1, u2, p=2).pow(2)
                    
                    # FIX #3: Add numerical safeguards
                    M = torch.clamp(M, max=1e6)
                    if torch.isnan(M).any() or torch.isinf(M).any():
                        self.logger.warning(f"NaN/Inf in OT cost matrix for {k1}-{k2}, skipping this pair")
                        continue
                        
                    a = (torch.ones(u1.shape[0], device=net.device) / u1.shape[0]).to(self.ot_dtype)
                    b = (torch.ones(u2.shape[0], device=net.device) / u2.shape[0]).to(self.ot_dtype)
                    M_cast = M.to(self.ot_dtype)
                    
                    try:
                        gamma = ot.sinkhorn(a, b, M_cast, reg=self.ot_reg, method='sinkhorn_stabilized')
                        current_pair_loss = torch.sum(gamma.to(M.dtype) * M)
                        cumulative_ot += current_pair_loss
                        pairs_count += 1
                    except Exception as e:
                        self.logger.warning(f"Sinkhorn failed for {k1}-{k2}: {e}")
                        continue
            if pairs_count > 0:
                ot_loss = cumulative_ot / pairs_count

        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos, n_neg = pos_mask.sum().item(), pos_mask.numel() - pos_mask.sum().item()
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device).scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        x_nll = {
            k: -net.u2x[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {k: D.kl_divergence(u[k], prior).sum(dim=1).mean() / x[k].shape[1] for k in net.keys}
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        pmsk = pmsk.T
        usamp_stack = torch.stack([usamp[k] for k in net.keys])
        pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        if self.normalize_u:
            usamp_mean = F.normalize(usamp_mean, dim=1)

        joint_cross_loss = torch.as_tensor(0.0, device=net.device)
        if self.lam_joint_cross:
            x_joint_cross_nll = {
                k: -net.u2x[k](
                    usamp_mean[m], vsamp[getattr(net, f"{k}_idx")],
                    xbch[k][m], None if l[k] is None else l[k][m]
                ).log_prob(x[k][m]).mean()
                for k, m in zip(net.keys, pmsk) if m.sum()
            }
            if x_joint_cross_nll:
                joint_cross_loss = sum(self.modality_weight[k] * nll for k, nll in x_joint_cross_nll.items())

        real_cross_loss = torch.as_tensor(0.0, device=net.device)
        if self.lam_real_cross:
            x_real_cross_nll = {}
            for k_tgt, m_tgt in zip(net.keys, pmsk):
                x_tgt_real_cross_nll = torch.as_tensor(0.0, device=net.device)
                for k_src, m_src in zip(net.keys, pmsk):
                    if k_src == k_tgt:
                        continue
                    m = m_src & m_tgt
                    if m.sum():
                        x_tgt_real_cross_nll += -net.u2x[k_tgt](
                            usamp[k_src][m], vsamp[getattr(net, f"{k_tgt}_idx")],
                            xbch[k_tgt][m], None if l[k_tgt] is None else l[k_tgt][m]
                        ).log_prob(x[k_tgt][m]).mean()
                x_real_cross_nll[k_tgt] = x_tgt_real_cross_nll
            if x_real_cross_nll:
                real_cross_loss = sum(self.modality_weight[k] * nll for k, nll in x_real_cross_nll.items())

        cos_loss = torch.as_tensor(0.0, device=net.device)
        if self.lam_cos:
            cos_losses = [
                1 - F.cosine_similarity(usamp_stack[i, m], usamp_mean[m]).mean()
                for i, m in enumerate(pmsk) if m.sum()
            ]
            if cos_losses:
                cos_loss = sum(cos_losses)

        vae_loss = (
            self.lam_data * x_elbo_sum + self.lam_graph * len(net.keys) * g_elbo +
            self.lam_sup * sup_loss + self.lam_joint_cross * joint_cross_loss +
            self.lam_real_cross * real_cross_loss + self.lam_cos * cos_loss
        )
        if self.alignment_method == "adversarial":
            # Multiply by alignment_weight to disable during pretraining
            gen_loss = vae_loss - (self.lam_align * alignment_weight) * dsc_loss 
        elif self.alignment_method == "ot":
            gen_loss = vae_loss + (self.lam_ot * alignment_weight) * ot_loss
        else:
            gen_loss = vae_loss

        losses = {
            "dsc_loss": dsc_loss, "ot_loss": ot_loss, "vae_loss": vae_loss,
            "gen_loss": gen_loss, "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            "joint_cross_loss": joint_cross_loss, "real_cross_loss": real_cross_loss,
            "cos_loss": cos_loss,
        }
        for k in net.keys:
            losses.update({f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]})
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses


# -------------------------------- Public API ----------------------------------


@logged
class SCGLUEModel(Model):

    r"""
    GLUE model for single-cell multi-omics data integration
    """

    NET_TYPE, TRAINER_TYPE = SCGLUE, SCGLUETrainer
    GRAPH_BATCHES, ALIGN_BURNIN_PRG, MAX_EPOCHS_PRG, PATIENCE_PRG, REDUCE_LR_PATIENCE_PRG = 32, 8.0, 48.0, 4.0, 2.0

    def __init__(
        self,
        adatas: Mapping[str, AnnData],
        vertices: List[str],
        latent_dim: int = 50,
        h_depth: int = 2,
        h_dim: int = 256,
        dropout: float = 0.2,
        shared_batches: bool = False,
        random_seed: int = 0,
        alignment_method: str = "adversarial",
    ) -> None:
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        self.alignment_method = alignment_method  # Store for later use
        torch.manual_seed(self.random_seed)

        g2v = sc.GraphEncoder(self.vertices.size, latent_dim)
        v2g = sc.GraphDecoder()
        self.modalities, idx, x2u, u2x, all_ct = {}, {}, {}, {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                data_config["rep_dim"] or len(data_config["features"]),
                latent_dim,
                h_depth=h_depth,
                h_dim=h_dim,
                dropout=dropout,
            )
            data_config["batches"] = (
                pd.Index([])
                if data_config["batches"] is None
                else pd.Index(data_config["batches"])
            )
            u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                len(data_config["features"]),
                n_batches=max(data_config["batches"].size, 1),
            )
            all_ct = all_ct.union(
                set()
                if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        
        # Only create discriminator for adversarial alignment
        if alignment_method == "adversarial":
            if shared_batches:
                all_batches = [modality["batches"] for modality in self.modalities.values()]
                ref_batch = all_batches[0]
                for batches in all_batches:
                    if not np.array_equal(batches, ref_batch):
                        raise RuntimeError(
                            "Batches must match when using `shared_batches`!"
                        )
                du_n_batches = ref_batch.size
            else:
                du_n_batches = 0
            du = sc.Discriminator(
                latent_dim,
                len(self.modalities),
                n_batches=du_n_batches,
                h_depth=h_depth,
                h_dim=h_dim,
                dropout=dropout,
            )
        else:
            # OT alignment - no discriminator needed
            du = None
        
        prior = sc.Prior()
        super().__init__(
            g2v, v2g, x2u, u2x, idx, du, prior,
            u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size),
        )

    def freeze_cells(self) -> None:
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
        self, source: "SCGLUEModel", submodule: Optional[str] = None
    ) -> None:
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(
        self,
        lam_data: float = 1.0,
        lam_kl: float = 1.0,
        lam_graph: float = 0.02,
        lam_align: float = 0.05,
        lam_sup: float = 0.02,
        alignment_method: str = "adversarial",
        lam_ot: float = 0.1,
        ot_reg: float = 0.1,
        ot_dtype: str = 'float32',
        gradient_clip_val: Optional[float] = None,
        normalize_u: bool = False,
        modality_weight: Optional[Mapping[str, float]] = None,
        lr: float = 2e-3,
        **kwargs,
    ) -> None:
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph, lam_align=lam_align,
            lam_sup=lam_sup, alignment_method=alignment_method, lam_ot=lam_ot,
            ot_reg=ot_reg, ot_dtype=ot_dtype, gradient_clip_val=gradient_clip_val,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs,
        )

    def fit(
        self,
        adatas: Mapping[str, AnnData],
        graph: nx.Graph,
        neg_samples: int = 10,
        val_split: float = 0.1,
        data_batch_size: int = 128,
        graph_batch_size: int = AUTO,
        align_burnin: int = AUTO,
        safe_burnin: bool = True,
        max_epochs: int = AUTO,
        patience: Optional[int] = AUTO,
        reduce_lr_patience: Optional[int] = AUTO,
        wait_n_lrs: int = 1,
        directory: Optional[os.PathLike] = None,
    ) -> None:
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train",
        )
        check_graph(
            graph, adatas.values(), cov="ignore", attr="error", loop="warn", sym="warn"
        )
        graph = GraphDataset(
            graph,
            self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True,
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG),
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG),
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG),
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG),
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data,
            graph,
            val_split=val_split,
            data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            align_burnin=align_burnin,
            safe_burnin=safe_burnin,
            max_epochs=max_epochs,
            patience=patience,
            reduce_lr_patience=reduce_lr_patience,
            wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory,
        )

    @torch.no_grad()
    def get_losses(
        self,
        adatas: Mapping[str, AnnData],
        graph: nx.Graph,
        neg_samples: int = 10,
        data_batch_size: int = 128,
        graph_batch_size: int = AUTO,
    ) -> Mapping[str, np.ndarray]:
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train",
        )
        graph = GraphDataset(
            graph,
            self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True,
        )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        return super().get_losses(
            data,
            graph,
            data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed,
        )

    @torch.no_grad()
    def encode_graph(
        self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        self.net.eval()
        graph = GraphDataset(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt), device=self.net.device
        )
        esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        v = self.net.g2v(eidx, enorm, esgn)
        if n_sample:
            return (
                torch.cat([v.sample((1,)).cpu() for _ in range(n_sample)])
                .permute(1, 0, 2)
                .numpy()
            )
        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
        self,
        key: str,
        adata: AnnData,
        batch_size: int = 128,
        n_sample: Optional[int] = None,
    ) -> np.ndarray:
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]], mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=False,
            persistent_workers=False,
        )
        result = []
        for x, xrep, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True,
            )[0]
            if n_sample:
                result.append(u.sample((n_sample,)).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def classify_data(
        self,
        key: str,
        adata: AnnData,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        self.net.eval()
        encoder = self.net.x2u[key]
        classifier = self.net.u2c
        data = AnnDataset(
            [adata], [self.modalities[key]], mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=False,
            persistent_workers=False,
        )
        result = []
        for x, xrep, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True,
            )[0]
            c = classifier(u.mean).softmax(dim=-1)
            result.append(c.detach().cpu())
        return pd.DataFrame(
            torch.cat(result).numpy(),
            index=adata.obs_names,
            columns=self.modalities[key]["cell_types"],
        )

    @torch.no_grad()
    def decode_data(
        self,
        source_key: str,
        target_key: str,
        adata: AnnData,
        graph: nx.Graph,
        target_libsize: Optional[Union[float, np.ndarray]] = None,
        target_batch: Optional[np.ndarray] = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, batch_size=batch_size)
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=False,
            persistent_workers=False,
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def __repr__(self) -> str:
        return (
            f"SCGLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )


@logged
class PairedSCGLUEModel(SCGLUEModel):

    r"""
    GLUE model for partially-paired single-cell multi-omics data integration
    """

    TRAINER_TYPE = PairedSCGLUETrainer

    def compile(
        self,
        lam_data: float = 1.0,
        lam_kl: float = 1.0,
        lam_graph: float = 0.02,
        lam_align: float = 0.05,
        lam_sup: float = 0.02,
        alignment_method: str = "adversarial",
        lam_ot: float = 0.1,
        ot_reg: float = 0.1,
        ot_dtype: str = 'float32',
        gradient_clip_val: Optional[float] = None,
        lam_joint_cross: float = 0.02,
        lam_real_cross: float = 0.02,
        lam_cos: float = 0.02,
        normalize_u: bool = False,
        modality_weight: Optional[Mapping[str, float]] = None,
        lr: float = 2e-3,
        **kwargs,
    ) -> None:
        super().compile(
            lam_data=lam_data,
            lam_kl=lam_kl,
            lam_graph=lam_graph,
            lam_align=lam_align,
            lam_sup=lam_sup,
            alignment_method=alignment_method,
            lam_ot=lam_ot,
            ot_reg=ot_reg,
            ot_dtype=ot_dtype,
            gradient_clip_val=gradient_clip_val,
            lam_joint_cross=lam_joint_cross,
            lam_real_cross=lam_real_cross,
            lam_cos=lam_cos,
            normalize_u=normalize_u,
            modality_weight=modality_weight,
            lr=lr,
            **kwargs,
        )
