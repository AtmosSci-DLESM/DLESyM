# DLESyM — AIMIP

This directory contains the files and configuration necessary to run **DLESyM** (Deep Learning Earth System Model) as a submission to the [AIMIP 2026](https://github.com/ai2cm/AIMIP) model intercomparison project.

DLESyM is a coupled atmosphere–ocean deep learning model for efficient simulation of the observed climate. This submission used the architecture and checkpoints described in [Cresswell-Clay et al. 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV001706).

**Model components:**
- **DLWP** (atmosphere): HEALPix-based U-Net for atmospheric state prediction
- **DLOM-gt** (ocean): A "ground-truth" model that coupled to DLWP but provides output prescribed by a forcing dataset. In this case, a processed version of the [standard SST forcing dataset published by Ai2](https://zenodo.org/records/17065758). 

**Forcing Data:**

Forcing data was retrived from Ai2-curated [Zenodo store](https://zenodo.org/records/17065758). For compatibility with our coupling scheme onthly data was resampled to daily using linear interpolation. 

**Submission Overview:**

Output from the following experiments are provided: `aimip`, `aimip-p2k`, and `aimip-p4k`. Initial submission will include key variables surface temperature (`tas`), temperature (`ta`) at 850hPa, and geopotential height (`zg`) at 1000, 500, and 250hPa. Monthly averages for the full historical period, and daily averages for the first 15 months are included. 

Five realizations were provided for each experiment resulting in 15 total simulations. Realizations were created using lagged initialization. Due to a limited data range in DLESyM's satellite derived outgoing longwave radiation data, simulations were started in 1983.

| Realization | Initialization |
|------|-------------|
| `r1` | 10/01/1983 |
| `r2` | 10/02/1983 |
| `r3` | 10/03/1983 |
| `r4` | 10/04/1983 |
| `r5` | 10/05/1983 |
---

## Directory Contents

| File | Description |
|------|-------------|
| `retrieve_zenodo.sh` | Script to download required standard forcing data from Zenodo|
| `preprocess_forcing.py` | process standardized forcing into a format compatible with DLESyM coupling scheme|
|`forcedforecast_1983-2025_5member.sh`| batch script for running AIMIP basic simulations|
|`forcedforecast_1983-2025_p2k.sh`| batch script for running AIMIP p2k simulations|
|`forcedforecast_1983-2025_p4k.sh`| batch script for running AIMIP p4k simulations|
|`cmortize_dlesym.py`| Routine for reformatting DLESyM output into CMIP-style output | 
|`aimip_validator.py`| class for checking output format |
|`test_submission.py`| test suite which invokes `cf-checker` and other basic validations of submission format| 

---

## Process

1. **Retrieve Forcing Data**: get standard forcing data using: 

      `python retrieve_zenodo.py`

2. **Prepare Forcing data**: prepare forcing data for ingestion into DLESyM atmosphere component: 
      `python preprocess_forcing.py`

3. **Run Experiments**: run requested experiments from AIMIP phase-1 call: 
      `bash forced_forecast_1983-2025_5member.sh`
      `bash forced_forecast_1983-2025_p2k.sh`      
      `bash forced_forecast_1983-2025_p4k.sh`

4. **"Cmortize" Output: enforce CMIP-style output: 
      `python cmortize_dlesym.py`

...and check that output satisfies expected structure
      `pytest test_submission.py -v`

Once tests are passed, we're ready to submit!

