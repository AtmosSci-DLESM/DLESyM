# DLESyM Zenodo Data Store

Model checkpoints, forecast files, training data, and cached analyses are hosted on the associated Zenodo repository (linked here). For all data relevent to this study I provide retrieval scripts here. Zenodo only hosts individual file sizes <50GB and does not elegently support hierarchical formats like Zarr. In these cases, retrieval scripts include processing steps so that data is ready for use in training, inference or evaluation. 

## File descriptions

`first5_1000yr_retrieval.sh`: downloads the first 5 years of a 1000-year simulation with DLESyM.

`last5_1000yr_retrieval.sh`: downloads the last 5 years of a 1000-year simulation with DLESyM. Note that because of limits of datetime[ns] encoding in the `step` dimension, times have been aliased to 2112-2117. 

`era5_z500_2017-2022_retrieval.sh`: era5 data interpolated into a 1-degree lat-lon mesh. Used as a reference fore seasonal cycles. 

`hpx64_lat-lon_ref_retrieval.sh`: download the reference lat-lon coordinates for each hpx64 point. 

`skill_forecast_retrieval.sh`: download 16-day forecasts for 105 initialization between years 2016 and 2017. 

`100yr_simulation_retrieval.sh`: download 100 year simulation output for the atmosphere.  

`forced_sst_1983-2017.sh`: download forced run of DLESyM's atmopsheric component drived by observed SSTs, 1983-2017.  

`dlesym_atmosphere_trainset.sh`: download full trainset for DLESyM's atmospheric component. This is also used in some evaluations routines. Includes concatenation of chunked files (made necessary by zenodo's file upload limit).

`analysis_cache_retrieval.sh`: downloads zipped analysis caches. Required for quick recreation of plots and curves presented in paper. Exclude for full replication of results. 