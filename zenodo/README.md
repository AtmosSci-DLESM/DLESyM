# DLESyM Zenodo Data Store

Model checkpoints, forecast files, training data, and cached analyses are hosted on the associated Zenodo repository (linked here). For all data relevent to this study I provide retrieval scripts here. Zenodo only hosts individual file sizes <50GB and does not elegently support hierarchical formats like Zarr. In these cases, retrieval scripts include processing steps so that data is ready for use in training, inference or evaluation. 

## File descriptions

`first5_1000yr_retrieval.sh` - downloads the first 5 years of a 1000 year simulation with DLESyM.

`last5_1000yr_retrieval.sh` - downloads the last 5 years of a 1000 year simulation with DLESyM. Note that because of limits of datetime[ns] encoding in the `step` dimension, times have been aliased to 2112-2117. 

`analysis_caches.sh` - downloads zipped analysis caches. Required for quick recreation of plots and curves presented in paper. Exclude for full replication of results. 

`era5_z500_2017-2022_retrieval.sh` - era5 data interpolated into a 1-degree lat-lon mesh. Used as a reference fore seasonal cycles. 