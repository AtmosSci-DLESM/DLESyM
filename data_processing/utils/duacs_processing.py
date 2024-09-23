import cdsapi
import zipfile
import numpy as np
import os
from datetime import datetime, timedelta

# Define a dictionary to hold parameters for data processing
EXAMPLE_RETREIVAL_PARAMS = {
    # Create the range of yeeears to retrieve data for
    "years": np.arange(1993, 1995),
    # Specify the output directory where the retrieved data will be saved
    "output_directory": "/home/gold2/dlwp/data/DUACS/raw_data",
    # Set the overwrite flag to False, indicating existing files should not be overwritten
    "overwrite": False,
}


def retrieve(params):
    print("Beginning to retrieve DUACS files...")

    # Keep track of which years were skipped
    skipped_years = []

    # create directory if it doesn't exist
    if not os.path.exists(params["output_directory"]):
        os.makedirs(params["output_directory"])

    def get_request_files(request):
        """
        Retrieves DUACS (Delayed-Time Level-4 sea surface height and derived variables) files for a given set of parameters.

        The function retrieves data for each year specified in the 'years' key of the params dictionary.
        It checks if the files already exist in the specified output directory and skips the retrieval if they do.
        The retrieved data is downloaded as a zip file and then extracted in the output directory.

        Parameters:
        params (dict): A dictionary containing the parameters for the data retrieval.
                    It should have the following keys:
                    - 'years': a list or array of years to retrieve data for
                    - 'output_directory': the directory to save the retrieved data in
                    - 'overwrite': a boolean indicating whether to overwrite existing files

        Returns:
        None
        """

        # internal function that turns a year into a list of days represented in the format YYYYMMDD
        def year_to_days(year):
            start_date = datetime.strptime(year, "%Y")
            end_date = datetime.strptime(str(int(year) + 1), "%Y")
            delta = timedelta(days=1)
            days = []
            while start_date < end_date:
                days.append(start_date.strftime("%Y%m%d"))
                start_date += delta
            return days

        filenames = [
            f"{params['output_directory']}/dt_global_twosat_phy_l4_{YYYYMMDD}_vDT2021.nc"
            for YYYYMMDD in year_to_days(request["year"])
        ]
        return filenames

    for year in params["years"]:
        c = cdsapi.Client()
        request = {
            "version": "vDT2021",
            "variable": "daily",
            "format": "zip",
            "year": f"{year}",
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
        }
        request_files = get_request_files(request)

        # check if any of the files already exist
        if np.any([os.path.exists(request_file) for request_file in request_files]):
            skipped_years.append(year)
            continue

        # retrieve
        c.retrieve(
            "satellite-sea-level-global",
            request,
            f"{params['output_directory']}/download.zip",
        )
        # unzip data
        with zipfile.ZipFile(
            f"{params['output_directory']}/download.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{params['output_directory']}/")

        # release memory
        del c
        os.remove(f"{params['output_directory']}/download.zip")

    if len(skipped_years) > 0 and not params["overwrite"]:
        print(
            f"The following years were skipped because some or all of the associated files already exist. To force downloading set params['overwrite'] to True."
        )
        print(f"{skipped_years}")
    print("Finished retirving requested DUACS files.")


import xarray as xr
from dask.diagnostics import ProgressBar

EXAMPLE_FIX_COORDS_PARAMS = {
    "variable_name": "adt",
    "input_file": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.nc",
    "output_file": "/home/disk/rhodium/dlwp/data/DUACS/dt_global_twosat_phy_l4_imputed_1993-2022_vDT2021_adt.pp.nc",
}


def fix_coords(params):
    # check that output file doesn't already exist
    if os.path.exists(params["output_file"]):
        print(
            f'Output file {params["output_file"]} already exists. Skipping coordinate fixing...'
        )
        return

    da = xr.open_dataset(params["input_file"], chunks={"time": 1})[
        params["variable_name"]
    ]
    da = da.roll(longitude=int(len(da.longitude.values) / 2), roll_coords=True)
    da = da.sortby("latitude", ascending=False)

    with ProgressBar():
        da.to_netcdf(params["output_file"])
    print("...Finished!")
    print(da)


if __name__ == "__main__":
    retrieve(EXAMPLE_RETREIVAL_PARAMS)
    fix_coords(EXAMPLE_FIX_COORDS_PARAMS)
