# DLESyM — AIMIP

This directory contains the files and configuration necessary to run **DLESyM** (Deep Learning Earth System Model) as a submission to the [AIMIP 2026] model intercomparison project.

## Model Overview

DLESyM is a coupled atmosphere–ocean deep learning model for efficient simulation of the observed climate. This submission uses the architecture and checkpoints described in [Cresswell-Clay et al. 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV001706).

**Model components:**
- **DLWP** (atmosphere): HEALPix-based U-Net for atmospheric state prediction
- **DLOM-gt** (ocean): A "ground-truth" model that coupled to DLWP but provides output prescribed by a forcing dataset. In this case, a processed version of the [standard SST forcing dataset published by Ai2](https://zenodo.org/records/17065758). 

---

## Directory Contents

| File | Description |
|------|-------------|
| `retrieve_forecing.sh` | Script to download required standard forcing data from Zenodo|

---

## Process

1. **Retrieve Forcing Data**: get standard forcing data using: 

      `python retrieve_zenodo.py`

2. **Prepare Forcing data**: prepare forcing data for ingestion into DLESyM atmosphere component: 
      `python preprocess_forcing.py`

3. run experiments
