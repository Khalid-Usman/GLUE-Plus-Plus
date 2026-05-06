r"""
Integration models
"""

import os
from pathlib import Path
from typing import Mapping, Optional

import dill
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

# Check for POT availability for Optimal Transport
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    # Don't raise here - check later when actually needed

from ..data import estimate_balancing_weight
from ..typehint import Kws
from ..utils import config, logged
from .base import Model
from .dx import integration_consistency
from .nn import autodevice
from .scclue import SCCLUEModel
from .scglue import PairedSCGLUEModel, SCGLUEModel


@logged
def configure_dataset(
    adata: AnnData,
    prob_model: str,
    use_highly_variable: bool = True,
    use_layer: Optional[str] = None,
    use_rep: Optional[str] = None,
    use_batch: Optional[str] = None,
    use_cell_type: Optional[str] = None,
    use_dsc_weight: Optional[str] = None,
    use_obs_names: bool = False,
) -> None:
    r"""
    Configure dataset for model training.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = (
            adata.var.query("highly_variable").index.to_numpy().tolist()
        )
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = (
            pd.Index(adata.obs[use_batch])
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )  
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = (
            pd.Index(adata.obs[use_cell_type])
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        ) 
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None

    # ========== FIX #22: VALIDATE DATA FOR PROB_MODEL ==========
    if prob_model in ("NB", "ZINB"):
        # Check the data matrix for negative values
        data_matrix = adata.X if use_layer is None else adata.layers.get(use_layer)
        
        if data_matrix is not None:
            # Handle different matrix types
            if hasattr(data_matrix, 'min'):
                min_val = data_matrix.min()
            else:
                import numpy as np
                import scipy.sparse as sp
                if sp.issparse(data_matrix):
                    min_val = data_matrix.data.min() if data_matrix.data.size > 0 else 0
                else:
                    min_val = np.min(data_matrix)
            
            if min_val < 0:
                error_msg = (
                    "\n\n"
                    + "="*80 + "\n"
                    + "DATA/MODEL MISMATCH DETECTED (Issue #22)\n"
                    + "="*80 + "\n"
                    + "\n"
                    + f"Problem: Your data contains NEGATIVE values (min={min_val:.3f})\n"
                    + f"         but you're using prob_model='{prob_model}'\n"
                    + "\n"
                    + "Negative Binomial (NB/ZINB) requires NON-NEGATIVE integers (counts).\n"
                    + "Negative values typically come from sc.pp.scale() or normalization.\n"
                    + "\n"
                    + "Solutions:\n"
                    + "  1. Use raw counts: configure_dataset(..., use_layer='counts')\n"
                    + "  2. Use different model: prob_model='Normal' for scaled data\n"
                    + "\n"
                    + "Example fix:\n"
                    + "  # DON'T do this:\n"
                    + "  sc.pp.scale(adata)\n"
                    + "  configure_dataset(adata, prob_model='NB')  # WRONG!\n"
                    + "\n"
                    + "  # DO this instead:\n"
                    + "  configure_dataset(adata, prob_model='NB', use_layer='counts')\n"
                    + "  # Or:\n"
                    + "  sc.pp.scale(adata)\n"
                    + "  configure_dataset(adata, prob_model='Normal')\n"
                    + "\n"
                    + "="*80 + "\n"
                )
                raise ValueError(error_msg)
            
            configure_dataset.logger.info(
                f"✓ Data validation passed for {prob_model}: min={min_val:.3f}"
            )
    # ===========================================================
    
    data_config["use_obs_names"] = use_obs_names
    adata.uns[config.ANNDATA_KEY] = data_config


def load_model(fname: os.PathLike) -> Model:
    r"""Load model from file"""
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  
    model.net.device = autodevice()  
    return model


@logged
def check_pot_available():
    if not POT_AVAILABLE:
        check_pot_available.logger.error(
            "POT (Python Optimal Transport) is required. "
            "Please install it using: pip install POT"
        )
        return False
    return True


@logged
def fit_SCGLUE(
    adatas: Mapping[str, AnnData],
    graph: nx.Graph,
    model: type = SCGLUEModel,
    skip_balance: bool = False,
    init_kws: Kws = None,
    compile_kws: Kws = None,
    fit_kws: Kws = None,
    balance_kws: Kws = None,
) -> SCGLUEModel:
    r"""
    Fit GLUE model with HetGAT and OT integration.
    
    This workflow includes:
    1. Pretraining on individual modalities (data reconstruction)
    2. Fine-tuning with Optimal Transport alignment
    """
    if not check_pot_available():
        raise ImportError("POT is required.")
    
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}

    fit_SCGLUE.logger.info("Pretraining SCGLUE model (HetGAT+OT)...")
    
    # Pretraining configuration
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": 10000, "safe_burnin": False})  # Fixed: use finite burnin
    
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = os.path.join(
            pretrain_fit_kws["directory"], "pretrain"
        )

    # Instantiate model
    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    
    # Compile and fit pretraining
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    # Balancing weight estimation (Optional)
    if not skip_balance:
        fit_SCGLUE.logger.info("Estimating balancing weight...")
        for k, adata in adatas.items():
            adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
        if init_kws.get("shared_batches"):
            use_batch = set(
                adata.uns[config.ANNDATA_KEY]["use_batch"] for adata in adatas.values()
            )
            use_batch = use_batch.pop() if len(use_batch) == 1 else None
        else:
            use_batch = None
        estimate_balancing_weight(
            *adatas.values(),
            use_rep=f"X_{config.TMP_PREFIX}",
            use_batch=use_batch,
            key_added="balancing_weight",
            **balance_kws,
        )
        for adata in adatas.values():
            adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
            del adata.obsm[f"X_{config.TMP_PREFIX}"]

    fit_SCGLUE.logger.info("Fine-tuning SCGLUE model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = os.path.join(
            finetune_fit_kws["directory"], "fine-tune"
        )

    # Fine-tuning phase
    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(**compile_kws)
    
    # Jitter random seed
    fit_SCGLUE.logger.debug(
        "Increasing random seed by 1 to prevent identical data order..."
    )
    finetune.random_seed += 1
    
    finetune.fit(adatas, graph, **finetune_fit_kws)
    
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune
