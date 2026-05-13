from aimip.precip_diagnosis.precip_pipeline import main as precip_pipeline
from omegaconf import OmegaConf

if __name__ == "__main__":

    # p4k r1
    precip_pipeline({
        "tag": "p4k_r1",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_4k_n01.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_4k_n01.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r1",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_4k_n01",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r1/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_4k_n01',
            'gpu': 1,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r1/',
            'tag': 'p4k_r1',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r1/precip_aimip_forced_forecast_1978-2025_p4k_r1.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 1,
            "experiment": "aimip-p4k",
        },
    })
    # p4k r2
    precip_pipeline({
        "tag": "p4k_r2",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_4k_n02.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_4k_n02.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r2",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_4k_n02",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r2/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_4k_n02',
            'gpu': 1,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r2/',
            'tag': 'p4k_r2',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r2/precip_aimip_forced_forecast_1978-2025_p4k_r2.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 2,
            "experiment": "aimip-p4k",
        },
    })
    # p4k r3
    precip_pipeline({
        "tag": "p4k_r3",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_4k_n03.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_4k_n03.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r3",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_4k_n03",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r3/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_4k_n03',
            'gpu': 1,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r3/',
            'tag': 'p4k_r3',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r3/precip_aimip_forced_forecast_1978-2025_p4k_r3.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 3,
            "experiment": "aimip-p4k",
        },
    })
    # p4k r4
    precip_pipeline({
        "tag": "p4k_r4",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury2/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_4k_n04.nc",
            "ocean_input": "/home/disk/mercury2/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_4k_n04.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r4",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_4k_n04",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r4/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_4k_n04',
            'gpu': 1,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r4/',
            'tag': 'p4k_r4',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r4/precip_aimip_forced_forecast_1978-2025_p4k_r4.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 4,
            "experiment": "aimip-p4k",
        },
    })
    # p4k r5
    precip_pipeline({
        "tag": "p4k_r5",
        "prep_precip_inputs_params": {
            "atmos_input": "/home/disk/mercury3/nacc/forecasts/aimip/atmos_aimip_forced_forecast_1978-2025_4k_n05.nc",
            "ocean_input": "/home/disk/mercury3/nacc/forecasts/aimip/ocean_aimip_forcing_1978-2025_4k_n05.nc",
            "dst_directory": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r5",
            "dataset_name": "precip-init_aimip_forced_forecast_1978-2025_4k_n05",
            "constants": {
                "lsm": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_lsm.nc",
                "z": "/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_topography.nc"
            },
            "scaling": OmegaConf.load("/home/disk/mercury2/nacc/AIMIP2026/DLESyM/training/configs/data/scaling/hpx64_1983-2017.yaml"),
            "overwrite_nc_cache": False,
        },
        "run_diagnosis_params": {
            'model_path': '/home/disk/mercury2/nacc/AIMIP2026/DLESyM/models/precip',
            'destination_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r5/',
            'data_name': 'precip-init_aimip_forced_forecast_1978-2025_4k_n05',
            'gpu': 1,
            'output_directory': '/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r5/',
            'tag': 'p4k_r5',
            'overwrite_forecast': False,
        }, 
        "cmortize_precip_params": {
            "forecast_file": "/home/disk/brass/nacc/forecasts/aimip/precip/p4k_r5/precip_aimip_forced_forecast_1978-2025_p4k_r5.nc",
            "output_dir": "/home/disk/mercury3/nacc/aimip_subission_1978",
            "r": 5,
            "experiment": "aimip-p4k",
        },
    })