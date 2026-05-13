from aimip.precip_diagnosis.precip_pipeline import main as precip_pipeline
from omegaconf import OmegaConf

if __name__ == "__main__":

    # historical r1
    precip_pipeline({
        "tag": "r1",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_n01.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_n01.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/r1",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_n01",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r1/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_n01',
            'gpu': 2,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r1/',
            'tag': 'r1',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/r1/precip_aimip_forced_forecast_1978-2025_r1.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 1,
            "experiment": "aimip",
        },
    })
    # historical r2
    precip_pipeline({
        "tag": "r2",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_n02.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_n02.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/r2",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_n02",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r2/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_n02',
            'gpu': 2,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r2/',
            'tag': 'r2',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/r2/precip_aimip_forced_forecast_1978-2025_r2.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 2,
            "experiment": "aimip",
        },
    })
    # historical r3
    precip_pipeline({
        "tag": "r3",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_n03.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_n03.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/r3",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_n03",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r3/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_n03',
            'gpu': 2,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r3/',
            'tag': 'r3',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/r3/precip_aimip_forced_forecast_1978-2025_r3.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 3,
            "experiment": "aimip",
        },
    })
    # historical r4
    precip_pipeline({
        "tag": "r4",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_n04.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_n04.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/r4",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_n04",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r4/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_n04',
            'gpu': 2,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r4/',
            'tag': 'r4',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/r4/precip_aimip_forced_forecast_1978-2025_r4.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 4,
            "experiment": "aimip",
        },
    })
    # historical r5
    precip_pipeline({
        "tag": "r5",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_n05.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_n05.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/r5",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_n05",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r5/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_n05',
            'gpu': 2,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/r5/',
            'tag': 'r5',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/r5/precip_aimip_forced_forecast_1978-2025_r5.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 5,
            "experiment": "aimip",
        },
    })
