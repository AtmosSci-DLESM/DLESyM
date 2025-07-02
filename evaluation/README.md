# Evaluating DLESyM
Code required to recreate the analysis in [our study](https://arxiv.org/abs/2409.16247) is stored here. This catalogue is organized by analysis script/notebook. For each, I provide a short description, identify significance to the study, and list the Zenodo-hosted data required to reproduce. For specific instructions on retrieving Zenodo-hosted data, see the `DLESyM/Zenodo` subdirectory. If you would like to recreate the inference experiments as well as the analyses, you will not need to retrieve forecast files from Zenodo. Instead see `DLESyM/inference` for instruction on recreating these experiments. 

For ease of replication, I have included some processed output from CMIP experiments as well as processed ERA5 reanalysis data. Full citation and acknowledgmenet of data sources is provided in our paper. 

| File | Description | Significance | Zenodo Retrievals | Notes |
|----------|-------------|----------|----------|----------|
|  `broken_ts_drify.py`    |   Plot zonally averaged Z500 seasonal cycle for 1000-year forecast and globally averaged timeseries.     |   Panels c-f of Figure 2 in manuscript      |   `DLESyM/zenodo/first5_1000yr_retrieval.sh`, `DLESyM/zenodo/last5_1000yr_retrieval.sh`, `DLESyM/zenodo/era5_z500_2017-2022_retrieval.sh`    |   Full 1000-year forecast is roughly 2 TB large and does not fit into Zenodo data repository. Instead the first 5 and last 5 years of the forecast are published. Last 5 forecast aliases `step` dimension to 2112-2117 due to limits of datetime[ns] encoding.    |
|   A2     |   B2     |   C2     |   D2     |   E2     |
|   A3     |   B3     |   C3     |   D3     |   E3     |

### Evaluation tips 

- References to forecast files and analysis caches will be relative and assume data the data is stored in the directory `./dlesym_zenodo/`.
