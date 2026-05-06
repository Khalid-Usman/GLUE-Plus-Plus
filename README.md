# GLUE++: Enhanced Variants of GLUE for Single-Cell Multi-Omics Integration

GLUE++ is an extension of the original GLUE framework for single-cell multi-omics integration. This repository contains multiple experimental and enhanced variants of the original GLUE architecture, focusing on improving graph representation learning and cross-modality alignment strategies.

The original GLUE framework was introduced by the Gao Lab for graph-linked unified embedding of single-cell multi-omics data.

Original GLUE Repository:  
https://github.com/gao-lab/GLUE

---

# Overview

The original GLUE framework primarily uses:

- Adversarial alignment for modality integration
- GCN (Graph Convolutional Network) for graph encoding

In **GLUE++**, several architectural improvements and experimental variants have been implemented, including:

## Enhancements Introduced

- Replacement of adversarial alignment with Optimal Transport (OT)-based alignment
- Replacement of GCN with:
  - Heterogeneous Graph Attention Networks (HetGAT)
  - Principal Neighbourhood Aggregation (PNA)
- Multiple modular implementations for comparative experimentation
- Separate source code for each model variant

These modifications are designed to explore stronger graph representation learning and improved cross-modality alignment for single-cell multi-omics integration tasks.

---

# Repository Structure

```bash
GLUE-Plus-Plus/
│
├── GLUE_GCN/                # Original GLUE-style GCN implementation
├── GLUE_HetGAT/             # GLUE with Heterogeneous GAT
├── GLUE_PNA/                # GLUE with Principal Neighbourhood Aggregation
├── GLUE_OT/                 # GLUE with Optimal Transport alignment
├── GLUE_OT_HetGAT/          # OT + HetGAT
├── GLUE_OT_PNA/             # OT + PNA
│
└── README.md
```

Each folder contains modified implementations of:

```bash
GLUE/scglue/models/
```

from the original GLUE repository.

---

# Installation

## 1. Clone Original GLUE Repository

```bash
git clone https://github.com/gao-lab/GLUE.git
cd GLUE
```

---

## 2. Create Environment

It is recommended to use the original GLUE environment setup.

```bash
conda env create -f env.yaml
conda activate scglue
```

Or install manually:

```bash
pip install scglue
```

For more details, refer to the official GLUE documentation.

---

# Using GLUE++ Variants

Each variant in this repository provides an alternative implementation of the `scglue/models/` module.

---

## Example: Running GLUE_HetGAT

### Step 1 — Clone Original GLUE

```bash
git clone https://github.com/gao-lab/GLUE.git
```

---

### Step 2 — Copy Variant Source Code

Copy the contents of:

```bash
GLUE_HetGAT/
```

into:

```bash
GLUE/scglue/models/
```

and replace the existing files.

---

### Step 3 — Run GLUE Normally

After replacement, run GLUE using the standard training pipeline and scripts provided in the original repository.

Example:

```bash
python train.py
```

or use the official notebooks/tutorials from GLUE.

---

# Available Variants

| Variant | Description |
|---|---|
| `GLUE_GCN` | Baseline implementation using Graph Convolutional Networks |
| `GLUE_HetGAT` | Replaces GCN with Heterogeneous Graph Attention Networks |
| `GLUE_PNA` | Uses Principal Neighbourhood Aggregation for graph representation learning |
| `GLUE_OT` | Replaces adversarial alignment with Optimal Transport-based alignment |
| `GLUE_OT_HetGAT` | Combines OT alignment with HetGAT |
| `GLUE_OT_PNA` | Combines OT alignment with PNA |

> Additional variants may be added in future updates.

---

# Key Research Ideas

## 1. Optimal Transport Alignment

The original GLUE framework uses adversarial learning for aligning latent spaces across modalities.

In GLUE++, Optimal Transport-based alignment strategies are explored to provide:

- More stable optimization
- Better distribution matching
- Reduced adversarial instability

---

## 2. Heterogeneous Graph Attention Networks (HetGAT)

HetGAT introduces attention mechanisms over heterogeneous biological graphs, enabling:

- Better feature interaction modeling
- Adaptive neighborhood weighting
- Improved biological relationship learning

---

## 3. Principal Neighbourhood Aggregation (PNA)

PNA enhances graph aggregation by combining multiple statistical aggregators and degree-scalers, potentially improving:

- Expressiveness
- Structural representation learning
- Robustness across graph topologies

---

# Compatibility

These implementations are designed as extensions/modifications of the original GLUE framework and are intended to remain compatible with:

- Existing GLUE preprocessing pipelines
- AnnData-based workflows
- Standard GLUE training procedures

---

# Citation

If you use the original GLUE framework, please cite the original paper:

```bibtex
@article{cao2022glue,
  title={GLUE: graph-linked unified embedding for single-cell multi-omics},
  author={Cao, Z. J. and others},
  journal={Nature Biotechnology},
  year={2022}
}
```

If you use GLUE++ in your research, please additionally cite this repository.

---

# Acknowledgements

This work is built upon the original:

- Gao Lab GLUE framework
- SCGLUE implementation
- PyTorch Geometric ecosystem

Original GLUE Repository:  
https://github.com/gao-lab/GLUE

---

# Disclaimer

This repository is an experimental research extension of GLUE and is intended for academic and research purposes only. Some variants may still be under active development and validation.

---

# Author

## Khalid Usman

PhD Researcher — Multi-Omics Integration & Graph Deep Learning

GitHub Repository:  
https://github.com/Khalid-Usman/GLUE-Plus-Plus
