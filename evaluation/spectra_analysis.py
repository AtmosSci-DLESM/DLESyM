import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

PARAMS_45N_1000th_year = {
    'skill_forecast':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_SkillForecast.nc',
    'forecast_1000':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year_last5_datetime.nc',
    'ref_file':'dlesym_zenodo/hpx64_ref_lat_lon.nc',
    'variable':'z500',
    'unit_conversion':9.81,
    'unit_label':'m$^2$ km',
    'lat':45,
    'title':'Z$_{500}$ Spectral Power at 45$^{\circ}$N',
    'leadtimes':[pd.Timedelta(d,'D') for d in [0,2]],
    'figure_file':'specta_era5_2day_1000th_year',
    'fcst_plot_params':[
        {'label':'ERA5',
         'color':'black',
         'linewidth':1.25,
         'alpha':1},
        {'label':'2 days',
         'color':'red',
         'linewidth':1.25,
         'alpha':1},
        {'label':'1000 years',
         'color':'green',
         'linewidth':1.25,
         'alpha':1},
    ],
}


PARAMS_45N_1000th_year_ws = {
    'skill_forecast':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_SkillForecast.nc',
    'forecast_1000':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year_last5_datetime.nc',
    'ref_file':'dlesym_zenodo/hpx64_ref_lat_lon.nc',
    'variable':'ws10',
    'unit_conversion':1,
    'unit_label':'m$^2$ s$^{-2}$ km',
    'lat':45,
    'title':'Windspeed Spectral Power at 45$^{\circ}$N',
    'show_legend':False,
    'leadtimes':[pd.Timedelta(d,'D') for d in [0,2]],
    'figure_file':'specta_era5_2day_1000th_year_ws',
    'fcst_plot_params':[
        {'label':'ERA5',
         'color':'black',
         'linewidth':1.25,
         'alpha':1},
        {'label':'2 days',
         'color':'red',
         'linewidth':1.25,
         'alpha':1},
        {'label':'1000 years',
         'color':'green',
         'linewidth':1.25,
         'alpha':1},
    ],
}

PARAMS_45N_1000th_year_t850 = {
    'skill_forecast':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_SkillForecast.nc',
    'forecast_1000':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year_last5_datetime.nc',
    'ref_file':'dlesym_zenodo/hpx64_ref_lat_lon.nc',
    'variable':'t850',
    'unit_conversion':1,
    'unit_label':'K$^2$ km',
    'lat':45,
    'title':'Windspeed Spectral Power at 45$^{\circ}$N',
    'show_legend':True,
    'leadtimes':[pd.Timedelta(d,'D') for d in [0,2]],
    'figure_file':'specta_era5_2day_1000th_year_t850',
    'fcst_plot_params':[
        {'label':'ERA5',
         'color':'black',
         'linewidth':1.25,
         'alpha':1},
        {'label':'2 days',
         'color':'red',
         'linewidth':1.25,
         'alpha':1},
        {'label':'1000 years',
         'color':'green',
         'linewidth':1.25,
         'alpha':1},
    ],
}
PARAMS_45N_1000th_year_olr = {
    'skill_forecast':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_SkillForecast.nc',
    'forecast_1000':'dlesym_zenodo/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_1000year_last5_datetime.nc',
    'ref_file':'dlesym_zenodo/hpx64_ref_lat_lon.nc',
    'variable':'olr',
    'unit_conversion':1,
    'unit_label':'W$^2$ m$^{-4}$ km',
    'lat':45,
    'title':'Windspeed Spectral Power at 45$^{\circ}$N',
    'show_legend':False,
    'leadtimes':[pd.Timedelta(d,'D') for d in [0,2]],
    'figure_file':'specta_era5_2day_1000th_year_olr',
    'fcst_plot_params':[
        {'label':'ERA5',
         'color':'black',
         'linewidth':1.25,
         'alpha':1},
        {'label':'2 days',
         'color':'red',
         'linewidth':1.25,
         'alpha':1},
        {'label':'1000 years',
         'color':'green',
         'linewidth':1.25,
         'alpha':1},
    ],
}


def get_closest_lat(target, lats):

    return lats.values[np.unravel_index(np.argmin(np.abs(lats.values-target)),lats.values.shape)] 

def get_lat_band(da, ref_lat, ref_lon, lat):
    

    print(f'Latitude requested was {lat}')
    closest = get_closest_lat(lat, ref_lat)
    boo = ref_lat.values == closest
    print(f'Using closest available lat: {closest} with {boo.sum()} points')
    
    # find the indices that will sort the latitude band by longitude 
    sorted_lon = ref_lon.values[boo].argsort()
    # initialize lat_band array and populate times with lat slices sorted by their lon 
    lat_band = np.empty(da.values.shape[0:2]+(len(sorted_lon),))
    print('Populating latitude band from HEALPix data...')
    times = tqdm(range(lat_band.shape[0]))
    steps = tqdm(range(lat_band.shape[1]),leave=False)
    for t in times:
        times.set_description(f'{str(da.time.values[t])[:13]}')
        for s in steps:
            steps.set_description(f'{str(s)}')
            lat_band[t,s,:] = da.values[t,s,:,:][boo][sorted_lon]
 
    # create dataarray to return 
    lat_band_da = xr.DataArray(
        data = lat_band,    
        dims = ['time','step','lon'],
        coords = dict(
            time=(['time'], da.time.values),
            step=(['step'], da.step.values),
            lon=(['lon'], ref_lon.values[boo][sorted_lon])
        ),
        attrs=dict(
            latitude=closest
        ))
    return lat_band_da

def get_nondimensional_wn(lon,lat):
    
    dx = (111.321*np.cos((np.pi/180)*lat))*np.abs(lon[0]-lon[1])
    return np.fft.rfftfreq(len(lon),dx)*len(lon)*dx

def get_zonal_wavelength(lon,lat):
    
    dx = (111.321*np.cos((np.pi/180)*lat))*np.abs(lon[0]-lon[1])
    return (len(lon)*dx)/get_nondimensional_wn(lon,lat)

def normalize_spectra(spectra, rectangular, lat):
 
    # normalization routine used to satisfy Parseval's relation 
    # taken from Durran et al. 2017 equation 13
    

    # Enforce Parseval's relation of transform unity and return.
    dx = (np.cos(np.deg2rad(lat))*111321)*np.mean(rectangular.lon[1:].values-rectangular.lon[:-1].values)
    N = len(rectangular.lon)
    kd = np.ones(len(spectra))
    kd[-1]=0
    return (dx/(np.pi*N*(1+kd)))*spectra**2

def get_spectra(lat_band, check_parseval_sat=False):
   
    spectra = []
    for t in lat_band.time:
        spectra.append(normalize_spectra(np.abs(np.fft.rfft(lat_band.sel(time=t))),lat_band.sel(time=t),lat_band.latitude))
    
    # check parseval's theorem
    if check_parseval_sat:
    # this expression of Parseval's relation is taken from Durran et al. 2017 equation 11 
        print('Departure from absolute satisfaction of Parsevals Theorem by leadtime:')
        dx = ((np.cos(np.deg2rad(lat_band.latitude))*111321)*(lat_band.lon[1]-lat_band.lon[0])).values
        L = dx*len(lat_band.isel(time=0))
        dk = 2*np.pi/L
        print(1-((1/L)*(dx*(lat_band.values)**2).sum(axis=1)/((np.array(spectra))*dk).sum(axis=1)))
    return np.array(spectra)

def main(
    skill_forecast,
    forecast_1000,
    ref_file,
    variable,
    unit_conversion,
    unit_label,
    lat,
    leadtimes,
    figure_file,
    title,
    fcst_plot_params,
    show_legend=True,
):
    
    # open forecast file and extract latitude band 
    skill_fcst = xr.open_dataset(skill_forecast)[variable]/unit_conversion
    fcsts_1000 = xr.open_dataset(forecast_1000)[variable]/unit_conversion
    ref_lat = xr.open_dataset(ref_file)['lat']
    ref_lon = xr.open_dataset(ref_file)['lon']
    lat_band_skill = get_lat_band(skill_fcst, ref_lat, ref_lon, lat)
    lat_band_1000 = get_lat_band(fcsts_1000, ref_lat, ref_lon, lat)
    xr.set_options(keep_attrs=True)
    # convert to height
    # lat_band = lat_band/9.81
    
    fig, ax = plt.subplots(figsize=(6,6))
    wavelength = get_zonal_wavelength(lat_band_skill.lon.values,get_closest_lat(lat,ref_lat))
    spectras = []
    # skill forecast spectra
    for i,lt in enumerate(leadtimes):
            spectras.append(get_spectra(lat_band_skill.sel(step=lt)).mean(axis=0))
    # 1000 year spectra
    lat_band_1000_last_year = lat_band_1000.squeeze().drop('time').rename({'step':'time'}).isel(time=np.arange(-4*366.25,-1,4).astype('int'))
    spectras.append(get_spectra(lat_band_1000_last_year).mean(axis=0))

    for i, spectra in enumerate(spectras):

        ax.plot(wavelength[:-1], spectra[:-1], **fcst_plot_params[i]) 
    # plot reference line 
    #ax.plot(wavelength,wavelength**ref_line_exp,alpha=.5,color='black',linestyle='dotted')
     
    # set up axis and labels 
    ax.invert_xaxis() 
    ax.grid(which='both',)
    ax.set_xscale('log')
    ax.set_xlabel('Wavelength (km)', fontsize=10)
    # settick labels to have fontsize 10
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_yscale('log')
    ax.set_ylabel('Power Spectral Density (' + unit_label + ')',fontsize=10)
    # ax.set_title(title)
    if show_legend:
        fig.legend(loc=(.15,.12), fontsize=10)
    fig.tight_layout()
    # save figure 
    fig.savefig(figure_file+'.pdf', dpi=300)
    fig.savefig(figure_file+'.png', dpi=300)

if __name__ == "__main__":
    

    main(**PARAMS_45N_1000th_year)
    # main(**PARAMS_45N_1000th_year_ws)
    # main(**PARAMS_45N_1000th_year_t850)
    # main(**PARAMS_45N_1000th_year_olr)
