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

Five realizations were provided for each experiment resulting in 15 total simulations. Realizations were created using lagged initialization. Due to a limited data range in DLESyM's satellite-derived outgoing longwave radiation (OLR) data, we use ERA5's `top_net_thermal_radiation` (TTR) field to initialize the simulations in 1978, as requested in the [AIMIP specifications] (https://github.com/ai2cm/AIMIP). The TTR field was fitted to OLR using an affine tranformation. Initializations were started on October 3, 1978 (extending to October 7th 1978) to allow for complete DLESyM initialization with available forcing data. 

| Realization | Initialization |
|------|-------------|
| `r1` | 10/03/1978 |
| `r2` | 10/04/1978 |
| `r3` | 10/05/1978 |
| `r4` | 10/06/1978 |
| `r5` | 10/07/1978 |
---

NOTE:

## Directory Contents

| File | Description |
|------|-------------|
| `retrieve_zenodo.sh` | Script to download required standard forcing data from Zenodo|
| `preprocess_forcing.py` | process standardized forcing into a format compatible with DLESyM coupling scheme|
|`forcedforecast_1978-2025_5member.sh`| batch script for running AIMIP basic simulations|
|`forcedforecast_1978-2025_p2k.sh`| batch script for running AIMIP p2k simulations|
|`forcedforecast_1978-2025_p4k.sh`| batch script for running AIMIP p4k simulations|
|`cmortize_dlesym.py`| Routine for reformatting DLESyM output into CMIP-style output | 
|`aimip_validator.py`| class for checking output format |
|`test_submission.py`| test suite which invokes basic validations of submission format| 
|`cfcheck.sh`| check for cf-compliant forecasts. |
|`precip_diagnosis/`| downstream pipeline for diagnosing precipitation from completed DLESyM simulations (see [Precipitation Diagnosis](#precipitation-diagnosis) below) |

---

## Process

1. **Retrieve Forcing Data:** get standard forcing data using: 

      `python retrieve_zenodo.py`

2. **Prepare Forcing data:** prepare forcing data for ingestion into DLESyM atmosphere component:  
      `python preprocess_forcing.py`

3. **Run Experiments:** run requested experiments from AIMIP phase-1 call:  
`bash forced_forecast_1978-2025_5member.sh`  
`bash forced_forecast_1978-2025_p2k.sh`  
`bash forced_forecast_1978-2025_p4k.sh`

4. **"Cmortize" Output:** enforce CMIP-style output:   
      `python cmortize_dlesym.py`

5. **Check output format:** check that output satisfies expected structure, variable names, etc...  
      `pytest test_submission.py -v`

      ...also check cf-compliant files. This requires use of a seperate checker:  
            `bash cfchecker.sh`

6. **Submit forecasts:** Once tests are passed, we're ready to submit!  
      `python submission_dkrz.py`

7. **Optional Check for validity of upload:**  
      `python verify_remote_submission.py`

**Note:** you'll need to obtain DKRZ credientials for uploads and dowloads. 

---

## Precipitation Diagnosis

DLESyM's atmosphere component (DLWP) does not predict precipitation as a prognostic variable. Instead, precipitation is diagnosed *after* a simulation has completed using a separate diagnostic model that takes DLWP atmospheric state as input. The `precip_diagnosis/` directory contains a three-stage pipeline that takes a finished DLESyM forecast and produces CMIP-style precipitation output (`pr`) for AIMIP submission.

The pipeline is orchestrated by `precip_pipeline.py`, which chains together three stages for a single (experiment, realization) pair:

1. **`prep_precip_inputs`** — repackages the raw atmosphere + ocean forecast `.nc` files into a single zarr dataset on the HPX64 grid, with the variable set, scaling, and constants (land–sea mask, topography) expected by the precip diagnosis model.
2. **`run_diagnosis`** — runs the trained precip diagnostic model (located at `models/precip`) on the prepared inputs to produce a precipitation forecast.
3. **`cmortize_precip`** — converts the raw precip forecast to CMIP-style output (correct units, time coordinates, metadata, daily and monthly averages) ready to be merged into the AIMIP submission.

Per-experiment driver scripts define the input/output paths and per-realization parameters for each of the three AIMIP experiments:

| File | Description |
|------|-------------|
| `precip_pipeline.py` | core pipeline that runs the three stages in sequence for a single configuration |
| `prep_precip_inputs.py` | stage 1: prepare zarr inputs from a completed DLESyM forecast |
| `run_diagnosis.py` | stage 2: run the precip diagnostic model on prepared inputs |
| `cmortize_dlesym_1978_precip.py` | stage 3: reformat diagnosed precip into CMIP-style output |
| `historical_precip_pipelines.py` | driver: runs the pipeline for all `aimip` (historical) realizations |
| `p2k_precip_pipelines.py` | driver: runs the pipeline for all `aimip-p2k` realizations |
| `p4k_precip_pipelines.py` | driver: runs the pipeline for all `aimip-p4k` realizations |

### Process

After the main simulations and `cmortize_dlesym.py` step above have completed, diagnose precipitation for each experiment:

```bash
python precip_diagnosis/historical_precip_pipelines.py
python precip_diagnosis/p2k_precip_pipelines.py
python precip_diagnosis/p4k_precip_pipelines.py
```

The CMIP-style `pr` files produced by stage 3 are written into the submission directory and can then be validated and uploaded using the same `test_submission.py` / `submission_dkrz.py` steps described above.

