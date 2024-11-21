#! env/bin/python3

"""
Takes a [hpx].nc forecast file and converts it into an according [ll].nc file.
"""

import argparse

import numpy as np
import xarray as xr
from tqdm import tqdm
import multiprocessing

import matplotlib.pyplot as plt

from remapper import HEALPixMapper
import istarmap  # https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423


def remap_parallel(mapper: HEALPixMapper, ds: xr.Dataset, vname: str, f_idx: int, s_idx: int) -> np.array:
    """
    Helper function to apply the mapping of individual samples (time steps) in parallel.

    :param mapper: The HEALPix mapper instance
    :param ds: The dataset containing the LatLon data
    :param vname: The string of the variable in the dataset to be remapped
    :param f_idx: The forecast start time index
    :param s_idx: The step in the forecast
    :return: A numpy array containing the data remapped to LatLon
    """
    return mapper.hpx2ll(ds.variables[vname][f_idx, s_idx].values)


def main(args):

    # Load .nc file in HEALPix format to get nside information and to initialize the remapper module
    fc_ds_hpx = xr.open_dataset(args.file_name)
    #fc_ds_hpx = fc_ds_hpx.isel({"time": slice(0, 2)})
    print(fc_ds_hpx)
    latitudes, longitudes =args.latitudes, args.longitudes
    nside = fc_ds_hpx.dims["height"]

    #exit()

    mapper = HEALPixMapper(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=nside,
        resolution_factor=args.resolution_factor
        )

    dims = [fc_ds_hpx.dims["time"], fc_ds_hpx.dims["step"], latitudes, longitudes]
    vname = "z500"
    
    if args.poolsize < 2:
        # Sequential sample mapping via for-loop

        # Allocate a (huge) array to store all samples (time steps) of the projected data
        fc_data_ll = np.zeros(dims, dtype=fc_ds_hpx.variables[vname])

        # Iterate over all samples and levels, project them to HEALPix and store them in the predictors array
        pbar = tqdm(fc_ds_hpx.coords["time"])
        for f_idx, forecast_start_time in enumerate(pbar):
            pbar.set_description("Remapping time steps")
            for s_idx, step in enumerate(fc_ds_hpx.coords["step"]):
                fc_data_ll[f_idx, s_idx] = mapper.hpx2ll(data=fc_ds_hpx.variables[vname][f_idx, s_idx].values,
                                                         visualize=args.visualize)
    else:
        # Parallel sample mapping with 'poolsize' processes
        
        # Collect the arguments for each remapping call
        arguments = []
        for f_idx, forecast_start_time in enumerate(fc_ds_hpx.coords["time"]):
            for s_idx, step in enumerate(fc_ds_hpx.coords["step"]):
                arguments.append([mapper, fc_ds_hpx, vname, f_idx, s_idx])

        # Run the remapping in parallel
        with multiprocessing.Pool(args.poolsize) as pool:
            print(f"Remapping time steps with {args.poolsize} processes in parallel")
            fc_data_ll = np.array(list(tqdm(pool.istarmap(remap_parallel, arguments), total=len(arguments))))
            pool.terminate()
            pool.join()
        fc_data_ll = np.reshape(fc_data_ll, dims)  # [(f s) lat lon] -> [f s lat lon]
    
    # Convert latitudes and longitudes from HEALPix to LatLon
    gt_ds = xr.open_dataset(f"/home/disk/quicksilver/nacc/Data/ERA5/1deg/1979-2022_era5_1deg_3h_geopotential_500.nc")
    lat, lon = gt_ds["latitude"], gt_ds["longitude"]

    # Set up coordinates and chunksizes for the LatLon dataset
    coords = {"time": fc_ds_hpx.coords["time"],
              "step": fc_ds_hpx.coords["step"],
              "lat": np.array(lat, dtype=np.int64),
              "lon": np.array(lon, dtype=np.int64)}
    
    # Build LatLon forecast dataset
    fc_ds_ll = xr.Dataset(coords=coords,
                          data_vars={vname: (list(coords.keys()), fc_data_ll)})
    
    # Load the according LatLon ground truth dataset
    print("Loading according LatLon ground truths")
    #gt_ds = xr.open_dataset(f"/home/disk/quicksilver/nacc/Data/ERA5/1deg/1979-2022_era5_1deg_3h_2m_temperature.nc")
    gt_data_ll = np.zeros_like(fc_data_ll)
    pbar = tqdm(fc_ds_hpx.coords["time"])
    for f_idx, forecast_start_time in enumerate(pbar):
        gt_data_ll[f_idx] = gt_ds.sel(
            {"time": forecast_start_time + fc_ds_hpx.coords["step"],
             "level": 500.0}
            )[list(gt_ds.keys())[0]].load()
    gt_ds_ll = xr.Dataset(coords=coords,
                          data_vars={vname: (list(coords.keys()), gt_data_ll)})
    
    fc_ds_ll.to_netcdf("baro_hpx64_forecasts_2016-2018.nc")
    gt_ds_ll.to_netcdf("baro_hpx64_ground_truths_2016_2018.nc")
    #exit()

    fc = fc_ds_ll.sel({"lat": slice(70, 20)})[vname].values
    gt = gt_ds_ll.sel({"lat": slice(70, 20)})[vname].values
    fc = fc_ds_ll[vname].values / 9.81
    gt = gt_ds_ll[vname].values / 9.81
    forecast_days = fc_ds_ll.step.values / np.timedelta64(24, 'h')

    rmses = np.sqrt(np.mean((fc-gt)**2, axis=(0, 2, 3)))
    np.save("baro/data/rmses_2017-2018.npy", rmses)
    np.save("baro/data/forecast_days_2017-2018.npy", forecast_days)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(forecast_days, np.mean(fc, axis=(0, 2, 3)), label="Forecast")
    ax[0].plot(forecast_days, np.mean(gt, axis=(0, 2, 3)), label="Ground truth")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(forecast_days, rmses)
    ax[1].grid()
    #ax[1].imshow(np.sqrt(np.square(fc[0, 1] - gt[0, 1])))
    #ax[0].imshow(fc[0, 1])
    #ax[1].imshow(gt[0, 1])
    plt.tight_layout()
    plt.savefig("rmse_baro_hpx64.pdf", format="pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Remap a HEALPix forecast file into LatLon.")
    parser.add_argument("-f", "--file-name", type=str, required=True,
                        help="Name or path to the HEALPix forecast file.")
    parser.add_argument("--latitudes", type=int, default=181,
                        help="Number of latitudes in the LatLon grid.")
    parser.add_argument("--longitudes", type=int, default=360,
                        help="Number of longitudes in the LatLon grid.")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Visualize the projected data.")
    parser.add_argument("--resolution-factor", type=float, default=1.0,
                        help="The multiplicative factor for the resolution of the projection. Use carefully.")
    parser.add_argument("--poolsize", type=int, default=20,
                        help="The size of the pool (number of parallel processes) used for data mapping.")

    run_args = parser.parse_args()
    main(args=run_args)
    print("Done.")
