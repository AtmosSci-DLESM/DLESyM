from evaluation.evaluators import EvaluatorHPX

# 1-in 1-out 24H 
# z500
_ = EvaluatorHPX(
    forecast_path = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    eval_variable = 'z500',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z500_ll.nc'
)
# # z1000
_ = EvaluatorHPX(
    forecast_path = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_1000.nc',
    eval_variable = 'z1000',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z1000_ll.nc'
)
# tau300-700
_ = EvaluatorHPX(
    forecast_path = '/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_tau300-700_new.nc',
    eval_variable = 'tau300-700',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/mercury5/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_tau300-700_ll.nc'
)


# 1-on 1-out 12H 
# z500
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    eval_variable = 'z500',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z500_ll.nc'
)
# z1000
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_1000.nc',
    eval_variable = 'z1000',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_z1000_ll.nc'
)
# tau300-700
_ = EvaluatorHPX(
    forecast_path = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit.nc',
    verification_path = '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_tau300-700_new.nc',
    eval_variable = 'tau300-700',
    on_latlon = True,
    poolsize = 20,
    ll_file=f'/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_1-to-1_seed0_12h-AR+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/atmos_100-year_JanInit_tau300-700_ll.nc'
)