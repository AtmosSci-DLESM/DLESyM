import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cfeature
import matplotlib.colors as colors
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import scipy.stats as stats
from types import SimpleNamespace
from xeofs.models import EOF  # pip install xeofs

def load_hgt_data(args):
    """
    Load geopotential data.
    calculate monthly anomalies and weight by latitude.
    select winter months (Nov-Apr) for northern hemisphere.
    select all months for southern hemisphere.
    Parameters:
    args.input_path: path to the input data
    args.era5_hgt: name of the ERA5 file
    args.var_name: variable name in the dataset
    args.start_time: start time for the data selection
    args.end_time: end time for the data selection
    args.region_box: bounding box for the region of interest
    Returns:
    weighted_polar_hgt: geopotential anomalies weighted by latitude
    polar_hgt_anom: geopotential anomalies
    ds_weights: weights for latitudes
    """
    ds_var = xr.open_dataset(args.input_path+args.era5_hgt)[args.var_name]

    [lon_min, lon_max, lat_min, lat_max] = args.region_box
    ds_var = ds_var.sel(time=slice(args.start_time, args.end_time)) \
                    .sel(latitude = slice(lat_max, lat_min), longitude = slice(lon_min, lon_max))

    # monthly mean
    polar_hgt = ds_var.resample(time='1M').mean(dim='time').squeeze()
    polar_hgt_anom = polar_hgt.groupby('time.month') - polar_hgt.groupby('time.month').mean()
    # As for north hemisphere, select winter months Nov-Apr
    # and for south hemisphere, select all months
    if lat_max > 0:
        sel_mon = (polar_hgt_anom['time.month'] >= 11) | (polar_hgt_anom['time.month'] <= 4)
        polar_hgt_anom = polar_hgt_anom.sel(time=sel_mon)

    # weight by latitude
    weight_lat = polar_hgt_anom.latitude.values
    weights = np.sqrt(np.cos(np.deg2rad(weight_lat)))
    weights[np.isnan(weights)] = 0.1
    ds_weights = xr.DataArray(weights, coords=[polar_hgt_anom.latitude], dims=['latitude'])
    weighted_polar_hgt = polar_hgt_anom * ds_weights
    return weighted_polar_hgt, polar_hgt_anom, ds_weights



def Annular_mode_EOF(weighted_polar_hgt, polar_hgt_anom, ds_weights, convert_to_hight=True):
    """
    Calculate leading EOF of geopotential.
    return Narthern Annular Mode (NAM) or Southern Annular Mode (SAM).
    Parameters:
    weighted_polar_hgt: geopotential anomalies with latitude weights
    polar_hgt_anom: geopotential anomalies
    ds_weights: weights for latitudes
    convert_to_hight: if True, convert to geopotential height
    Returns:
    reg_hgt: regression map of geopotential height
    scores: normalized PC time series
    expvar_ratio: explained variance ratio
    """
    model = EOF(n_modes=1)
    model.fit(weighted_polar_hgt, dim="time")
    expvar_ratio = model.explained_variance_ratio()
    components = model.components()
    scores = model.scores(normalized=False) #normalized=False
    print("Leading EOF mode: ")
    print("Relative: ", (expvar_ratio * 100).round(1).values)
    components.sortby("latitude", ascending=False)
    components[0] = -components[0]/ds_weights
    scores[0] = -scores[0]
    scores = (scores- scores.mean(dim='time'))/scores.std(dim='time')
    def linregress_func(y, scores):
        return stats.linregress(scores, y)
    slope, intercept, r_value, p_value, std_err = np.apply_along_axis(linregress_func, 0, polar_hgt_anom.values, scores)
    if convert_to_hight:
        reg_hgt = slope/9.8 # convert to geopotential height
    else:
        reg_hgt = slope
    reg_hgt = xr.DataArray(reg_hgt, coords=[polar_hgt_anom.latitude, polar_hgt_anom.longitude], dims=['latitude', 'longitude'])
    del scores.attrs['solver_kwargs']
    del expvar_ratio.attrs['solver_kwargs']
    return reg_hgt, scores, expvar_ratio


def plot_regression_map(reg_hgt, expvar_ratio, region_box, output_file=None):
    """
    Plot regression map of geopotential height.
    Parameters:
    reg_hgt: regression map of geopotential height
    expvar_ratio: explained variance ratio
    region_box: bounding box for the region of interest
    output_file: output file name
    """
    [lon_min, lon_max, lat_min, lat_max] = region_box
    if lat_max > 0:
        projection = ccrs.NorthPolarStereo(central_longitude=0.0)
    else:
        projection = ccrs.SouthPolarStereo(central_longitude=180.0)
    fig = plt.figure(figsize=(8,6))
    ax = fig.subplots(subplot_kw={'projection': projection})
    ax.add_feature(cfeature.COASTLINE,lw=0.5,color='k')
    gl=ax.gridlines(draw_labels=False,linestyle=":",linewidth=0.3 ,x_inline=False, y_inline=False,color='k')
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    cycle_components, cycle_lon = add_cyclic_point(reg_hgt, coord = reg_hgt.longitude)
    cnf = ax.contourf(cycle_lon, reg_hgt.latitude, cycle_components,
                        cmap='bwr', 
                        levels=np.arange(-50,50+5,5),
                        extend='both',
                        transform = ccrs.PlateCarree())
    cbar = plt.colorbar(cnf, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.text(0.95, 0.95, '%.1f%%' %(expvar_ratio*100), fontsize=11, 
                    ha='right', transform=ax.transAxes)
    if output_file is not None:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.show()
    return []


if __name__ == "__main__":
    args = SimpleNamespace()

    Annular_mode = 'NAM' # 'NAM' or 'SAM' 

    if Annular_mode == 'NAM':
        variable = 'z1000'
        lon_min = 0; lon_max = 360; lat_min = 20; lat_max = 90
    elif Annular_mode == 'SAM':
        variable = 'z500'
        lon_min = 0; lon_max = 360; lat_min = -90; lat_max = -20

    args.input_path = '/home/disk/rhodium/dlwp/data/era5/1deg/'
    args.era5_hgt = f'era5_1950-2022_3h_1deg_{variable}.nc'; args.var_name = 'z'
    output_EOF = f'{Annular_mode}_{variable}_ERA5.nc'
    output_regression_map = f'{Annular_mode}_{variable}_ERA5_regression_map.png'
    # select time period
    start_year = 1970; end_year = 2010
    args.start_time = np.datetime64('%d-01-01T00'%start_year)
    args.end_time = np.datetime64('%d-12-31T00'%end_year)
    # select region
    args.region_box = [lon_min, lon_max, lat_min, lat_max]

    # load data
    weighted_polar_hgt, polar_hgt_anom, ds_weights = load_hgt_data(args)
    # calculate leading EOF
    reg_hgt, scores, expvar_ratio = Annular_mode_EOF(weighted_polar_hgt, polar_hgt_anom, ds_weights)
    # save EOF results
    EOF_out = xr.Dataset({'reg_hgt': reg_hgt, 'scores': scores, 'expvar_ratio': expvar_ratio})
    EOF_out.to_netcdf(output_EOF)
    print("EOF results saved to: ", output_EOF)
    # plot regression map
    plot_regression_map(reg_hgt, expvar_ratio, args.region_box, output_regression_map)





