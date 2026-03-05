import numpy as np
import datetime
from tqdm import tqdm
import xarray as xr
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from itertools import chain
# from global_land_mask import globe
from scipy.ndimage import minimum_filter
from scipy.interpolate import RectBivariateSpline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from types import SimpleNamespace


def load_hpx(input_file, varname, args):
    """
    Load hpx data
    dimensions: (time, latitude, longitude)
    :param input_file: the input file name
    :param varname: the variable name
    :param args: the arguments
        include start_time, end_time, region_box
    return
    pacific_z1000: the hpx data over the West Pacific region
    day_interval: steps per day (int). if step=6h, day_interval=4.
    """
    ds_z1000 = xr.open_dataset(input_file) # 6h or daily data
    print(ds_z1000)
    ds_z1000 = change_hpx_coords(ds_z1000)
    print(ds_z1000)
    time = ds_z1000.time
    day_interval = (np.timedelta64(1,'D') / (time[1] - time[0]).astype('timedelta64[h]')).astype(int).values # 6h interval, 4 steps per day
    ds_z1000 = ds_z1000.sel(time = slice(args.start_time, args.end_time))[varname].squeeze() # last 30 years
    # select the West Pacific region
    [lon_min, lon_max, lat_min, lat_max] = args.region_box
    pacific_z1000 = ds_z1000.sel(latitude = slice(lat_max, lat_min), longitude = slice(lon_min, lon_max))

    return pacific_z1000, day_interval

def change_hpx_coords(ds_hgt):
    """
    Change HPX dataset coordinates for convenient processing.
    Parameters:
    ds_hgt: input HPX dataset.
    Returns:
    ds_hgt: dataset with changed coordinates
    """
    time = ds_hgt.time
    step = ds_hgt.step
    val_time = (time[0]+step).squeeze()
    # change the coordinate 'step' to 'val_time'
    ds_hgt = ds_hgt.assign_coords(step=val_time.values)
    ds_hgt = ds_hgt.rename({'step':'val_time'})
    # rename the coordinate 'val_time' to 'time' for convenient
    ds_hgt = ds_hgt.rename({'time':'int_time'})
    ds_hgt = ds_hgt.rename({'val_time':'time'})
    ds_hgt = ds_hgt.rename({'lon':'longitude'})
    ds_hgt = ds_hgt.rename({'lat':'latitude'})
    return ds_hgt

def cal_tau_clim(pacific_tau):
    """
    Calculate the climatological tau300-700 data over the West Pacific region
    :param pacific_tau: the forecast/era tau300-700 data over the West Pacific region
    return 
    tau_clim: the climatological tau300-700 data over the West Pacific region
    """
    # calculate the climatological tau300-700 data over the West Pacific region
    # tau_clim = pacific_tau.mean(dim='latitude').mean(dim='longitude')\
    #                     .groupby('time.dayofyear').mean(dim='time')
    pacific_tau_m = pacific_tau.mean(dim='latitude').mean(dim='longitude')
    ds = pd.DataFrame(pacific_tau_m)
    ds['tau'] = pd.Series(pacific_tau_m)
    ds['time'] = pd.to_datetime(pacific_tau_m.time)
    # Extract the month and day from the 'time' column
    ds['month'] = ds['time'].dt.month
    ds['day'] = ds['time'].dt.day
    ds['hour'] = ds['time'].dt.hour
    # fake year, just for convenience when grouping the data
    ds['year'] = 2000
    ds['date'] = pd.to_datetime(ds[['year','month', 'day', 'hour']])
    # tau.assign_coords(time=ds['date'])
    tau_clim = ds.groupby('date').mean()
    return tau_clim['tau']


def find_TC_tracks(args, higher_res = 0.05):
    """
    Find the tropical cyclones tracks
    :param args.pacific_z1000: the forecast/era z1000 data over the West Pacific region
    :param args.pacific_tau: the forecast/era tau300-700 data over the West Pacific region
    :param args.lsm: the land-sea mask
    :param args.tau_clim: the climatological tau300-700 data over the West Pacific region
    :param args.start_time: the start time of the forecast/era
    :param args.end_time: the end time of the forecast/era
    :param args.day_interval: steps per day (int). if step=6h, day_interval=4.
    :param higher_res: the higher resolution to interpolate the z1000 data, to find the local minimum. default is 0.05 degree.
    return 
    track_lat_z1000: the latitude of the tracks
    track_lon_z1000: the longitude of the tracks
    track_time_z1000: the time index of the tracks
    """
    # # the threshold of z1000 (HPX & ERA5): rank 1 to 4
    # approximate values of 1th, 0.1th, 0.01th, 0.001th percentiles of z1000
    z1000_th = [-100, -1000, -2000, -3000]

    # higher resolution for tracks
    [lon_min, lon_max, lat_min, lat_max] = args.region_box
    new_lat = np.arange(lat_min, lat_max+higher_res, higher_res)
    new_lon = np.arange(lon_min, lon_max+higher_res, higher_res)

    # find the target time index
    start_time_index = np.where(args.pacific_z1000.time >= args.start_time)[0][0]
    end_time_index = np.where(args.pacific_z1000.time <= args.end_time)[0][-1]

    track_lon_z1000={}  # location of the min(z1000) that exceeds the threshold
    track_lat_z1000={}
    track_time_z1000={}  # time index of the min(z1000)

    print('starting to find tracks')
    for ith in tqdm(range(len(z1000_th))): # loop over the rank
        threshold_z1000 = z1000_th[ith]
        # the location and time index of the min(z1000) for each rank
        track_lon = {}  
        track_lat = {}
        track_time = {}
        count = 1
        previous_track_indices = []
        for itime in tqdm(range(int(start_time_index), int(end_time_index) + 1)):
            outmap = args.pacific_z1000[itime, ...].to_numpy()
            outmap_tau = args.pacific_tau[itime, ...].to_numpy()
            ilocs = np.unravel_index(outmap.argmin(), outmap.shape) 
            # if the min(z1000) is less than the threshold, and the tau300-700 is larger than the climatological value, otherwise skip
            if np.min(outmap) < threshold_z1000 and outmap_tau[ilocs]>args.tau_clim[itime%(366*args.day_interval)]:
                # interpolate the z1000 to higher resolution
                rbs = RectBivariateSpline(args.pacific_z1000.latitude[::-1], args.pacific_z1000.longitude, outmap[::-1,:])
                outmap_high = rbs(new_lat, new_lon)
                # find local minimum in a 9x9 window
                filtered_output = minimum_filter(outmap_high, size=9*1/higher_res)  
                min_coords = np.where(outmap_high == filtered_output)
                for imin in range(len(min_coords[0])):
                    ilat = min_coords[0][imin]
                    ilon = min_coords[1][imin]
                    if outmap_high[ilat, ilon] < threshold_z1000:
                        if count == 1: 
                            ## if the track starts from the land, skip it
                            #if globe.is_ocean(new_lat[ilat], new_lon[ilon]):
                            if args.lsm.sel(latitude=new_lat[ilat], longitude=new_lon[ilon], method='nearest') <= 0.5:
                                track_lat['track' + str(count)] = [new_lat[ilat]]
                                track_lon['track' + str(count)] = [new_lon[ilon]]
                                track_time['track' + str(count)] = [itime]
                                previous_track_indices.append(count)
                                count += 1
                        else:
                            is_continuous = False # check if the track is continuous
                            for prev_index in previous_track_indices:
                                last_lat = track_lat['track' + str(prev_index)][-1]
                                last_lon = track_lon['track' + str(prev_index)][-1]
                                # check if the point is continuous with the previous track
                                # if the distance between the last point and the new point is less than 3 degree
                                # and the time difference is less than 2 days (if the time interval is 6h)
                                if (
                                    abs(track_time['track' + str(prev_index)][-1] - itime) <= args.day_interval * 2
                                    and abs(last_lat - new_lat[ilat]) <= 3
                                    and abs(last_lon - new_lon[ilon]) <= 3
                                ):
                                    track_lat['track' + str(prev_index)].append(new_lat[ilat])
                                    track_lon['track' + str(prev_index)].append(new_lon[ilon])
                                    track_time['track' + str(prev_index)].append(itime)
                                    previous_track_indices.append(prev_index)
                                    is_continuous = True
                                    break
                            # if the track is not continuous, create a new track
                            if not is_continuous:
                                #if globe.is_ocean(new_lat[ilat], new_lon[ilon]):
                                if args.lsm.sel(latitude=new_lat[ilat], longitude=new_lon[ilon], method='nearest') <= 0.5:
                                    track_lat['track' + str(count)] = [new_lat[ilat]]
                                    track_lon['track' + str(count)] = [new_lon[ilon]]
                                    track_time['track' + str(count)] = [itime]
                                    previous_track_indices.append(count)
                                    count += 1

        sys.stdout.flush()
        print('Rank %d: %d tracks found' % (ith + 1, len(track_lat)))

        track_lat_z1000['rank' + str(ith)] = track_lat
        track_lon_z1000['rank' + str(ith)] = track_lon
        track_time_z1000['rank' + str(ith)] = track_time

    return track_lat_z1000, track_lon_z1000, track_time_z1000


def filter_tracks(track_lat_z1000, track_lon_z1000, track_time_z1000):
    """
    Filter short tracks (less than 3 points)
    """
    for ith in range(len(track_time_z1000)):
        rank = 'rank%d'%ith
        for itrack in range(len(track_time_z1000[rank])):
            track_name = 'track%d' %(itrack+1)
            itrack_point = track_time_z1000[rank][track_name]
            if len(itrack_point) < 3: # filter the short tracks
                track_lat_z1000[rank].pop(track_name); track_lon_z1000[rank].pop(track_name)
                track_time_z1000[rank].pop(track_name)
                continue
    return track_lat_z1000, track_lon_z1000, track_time_z1000


def cal_TC_freq(pacific_z1000, track_time_z1000):
    """
    Calculate the frequency of tropical cyclones
    :param pacific_z1000: the z1000 data over the West Pacific region
    :param track_time_z1000: the time index of the extreme z1000
    return
    TC_freq: the frequency of tropical cyclones (dayofyear)
    """
    TC_freq = {} # TC per day per year
    for ikey in track_time_z1000.keys():
        time_index = list(track_time_z1000[str(ikey)].values())
        time_index = list(chain(*time_index)) # flatten the list
        # the time array for each rank
        rank_time_z1000 = pacific_z1000.time[time_index]
        # Convert the 'rank_time_z1000' variable to a pandas DataFrame
        df_rank_time_z1000 = pd.DataFrame(rank_time_z1000)
        df_rank_time_z1000['time'] = pd.to_datetime(rank_time_z1000)
        # Extract the month and day from the 'time' column
        df_rank_time_z1000['month'] = df_rank_time_z1000['time'].dt.month
        df_rank_time_z1000['day'] = df_rank_time_z1000['time'].dt.day
        # fake year, just for convenience when grouping the data
        df_rank_time_z1000['year'] = 2000
        df_rank_time_z1000['date'] = pd.to_datetime(df_rank_time_z1000[['year', 'month', 'day']])
        # Group the data by month and day, and count the occurrences
        occurrences_per_day = df_rank_time_z1000.groupby('date').size()
        occurrences_per_day = occurrences_per_day.asfreq('D', fill_value=0) # use asfreq to fill the missing days with 0
        TC_freq[str(ikey)] = occurrences_per_day
        del df_rank_time_z1000, occurrences_per_day
    return TC_freq


def plot_TC_freq(TC_freq, nyear, day_interval, output_file=None):
    """
    Plot the frequency of tropical cyclones
    :param TC_freq: the frequency of tropical cyclones
    :param nyear: the total number of years
    :param day_interval: forecast steps per day (int). if step=6h, then day_interval=4.
    :param output_file (optional): the output file of the plot
    """
    fig = plt.figure(figsize=(8, 6))
    colors = ['yellow','orange','orangered','red']
    print('\nStorms per year:')

    for ii,ikey in enumerate(TC_freq.keys()):
        # get the date and frequency of TC
        dates = TC_freq[str(ikey)].index
        frequency = TC_freq[str(ikey)].values/nyear/day_interval # Stormes per year
        plt.fill_between(dates, frequency, color=colors[ii], alpha=0.6)
        print('Rank %d:   %d'%(ii+1, round(frequency.sum())))

    plt.legend(['Rank1','Rank2','Rank3','Rank4'], loc='upper left')
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')
    plt.gca().xaxis.set_major_locator(months)
    plt.gca().xaxis.set_major_formatter(months_fmt)
    plt.ylabel('Storms per year')
    plt.xlim([datetime.date(2000, 1, 1), datetime.date(2000, 12, 31)])
    plt.ylim([0, 1.2])
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', dpi=150)
    return []



def plot_tracks(track_lat_z1000, track_lon_z1000, region_box, output_file=None):
    """
    Plot the tracks of tropical cyclones
    :param track_lat_z1000: the latitude of the tracks
    :param track_lon_z1000: the longitude of the tracks
    :param region_box: the region of the forecast/era
    :param output_file (optional): the output file of the plot
    """
    colors = ['yellow','orange','orangered','red']

    proj = ccrs.Mercator(globe=None)
    fig = plt.figure(figsize=(8,6))
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    ax.add_feature(cfeature.COASTLINE,lw=0.5,color='k')
    gl=ax.gridlines(draw_labels=False,linestyle=":",linewidth=0.3 ,x_inline=False, y_inline=False,color='k')

    for ith in range(len(track_lat_z1000)):
        rank = 'rank%d'%ith
        for track_name in track_lat_z1000[rank].keys():
            if len(track_lat_z1000[rank][track_name]) < 5: # filter the short tracks, just for plotting
                continue
            ax.plot(track_lon_z1000[rank][track_name], track_lat_z1000[rank][track_name], '.',
                        color=colors[ith], markersize=3, alpha=0.5, transform=ccrs.PlateCarree())
            
            ax.plot(track_lon_z1000[rank][track_name], track_lat_z1000[rank][track_name], '-',
                            color=colors[ith], linewidth=1.2, alpha=0.5, transform=ccrs.PlateCarree())

    legend_elements = [Line2D([0], [0], marker='.', color='w', label='Rank 1', markerfacecolor=colors[0], markersize=15),
                        Line2D([0], [0], marker='.', color='w', label='Rank 2', markerfacecolor=colors[1], markersize=15),
                        Line2D([0], [0], marker='.', color='w', label='Rank 3', markerfacecolor=colors[2], markersize=15),
                        Line2D([0], [0], marker='.', color='w', label='Rank 4', markerfacecolor=colors[3], markersize=15)]
    ax.legend(handles=legend_elements, loc='lower right')

    [lon_min, lon_max, lat_min, lat_max] = region_box
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # set the ticks
    lat_intv = 5; lon_intv = 10
    ax.set_xticks(np.arange(lon_min+lon_intv, lon_max, lon_intv), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(lon_min+lon_intv, lon_max, lon_intv), fontsize=10)
    ax.set_yticks(np.arange(lat_min+lat_intv, lat_max, lat_intv), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.arange(lat_min+lat_intv, lat_max, lat_intv), fontsize=10)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', dpi=150)
    return []

def main(
    input_dir,
    z1000_file,
    tau_file,
    output_prefix,
):
    
    args = SimpleNamespace()
    # path of the input/output files
    # hpx data: z1000 and tau300-700 (6h or daily data)
    # input_path = '/home/disk/rhodium/dlwp/data/hpx/1deg/'
    input_path = input_dir
    hpx_z1000 = z1000_file; varname_z1000 = 'z1000'
    hpx_tau = tau_file; varname_tau = 'tau300-700'
    # select the time period
    start_year = 2085; end_year = 2114
    args.start_time = np.datetime64('%d-01-01'%start_year)
    args.end_time = np.datetime64('%d-12-31'%end_year)
    nyear = (args.end_time - args.start_time).astype('timedelta64[Y]').astype(int)+1 # total 30 years
    # West Pacific region
    lon_min = 100; lon_max = 160; lat_min = 5; lat_max = 35
    args.region_box = [lon_min, lon_max, lat_min, lat_max]
    # land-sea mask. better to choose less than 0.25 degrees. 1: land, 0: sea
    args.lsm = xr.open_dataset('/home/disk/rhodium/dlwp/data/era5/0.25deg/1979-2021_era5_0.25deg_3h_land_sea_mask.nc')['lsm']

    # load hpx data
    args.pacific_z1000, args.day_interval = load_hpx(input_path+hpx_z1000, varname_z1000, args)
    args.pacific_tau, _ = load_hpx(input_path+hpx_tau, varname_tau, args)

    # calculate the climatological tau300-700 data over the West Pacific region
    args.tau_clim = cal_tau_clim(args.pacific_tau)

    # find the tropical cyclones tracks
    track_lat_z1000, track_lon_z1000, track_time_z1000 = find_TC_tracks(args)
    # filter the short tracks
    track_lat_z1000, track_lon_z1000, track_time_z1000 = filter_tracks(track_lat_z1000, track_lon_z1000, track_time_z1000)
    # calculate the frequency of tropical cyclones
    TC_freq = cal_TC_freq(args.pacific_z1000, track_time_z1000)

    # save the tracks & frequency
    output_tracks_file = '_%4d_%4d.npz'%(start_year, end_year)
    np.savez(output_prefix+output_tracks_file, track_lat_z1000=track_lat_z1000, track_lon_z1000=track_lon_z1000,\
                                                track_time_z1000=track_time_z1000, TC_freq=TC_freq)

    # plot the frequency of tropical cyclones
    output_freq = '_freq_hpx.png'
    plot_TC_freq(TC_freq, nyear, args.day_interval, output_file=output_prefix+output_freq)

    # plot the tracks of tropical cyclones
    output_tracks = 'tracks_%4d_%4d.png'%(start_year, end_year)
    plot_tracks(track_lat_z1000, track_lon_z1000, args.region_box, output_file=output_prefix+output_tracks)

if __name__ == '__main__':

    main(
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        z1000_file = 'atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z1000_ll.nc',
        tau_file = '/home/disk/rhodium/bowenliu/remap/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_tau300-700_ll.nc',
        output_prefix = './scratch/example',
    )