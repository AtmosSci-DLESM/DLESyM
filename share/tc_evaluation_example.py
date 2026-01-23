from evaluation.TC_freq_hpx import main as tc_main

params_example = dict(
        # directory of input file
        input_dir = '/home/disk/rhodium/nacc/forecasts/hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart/',
        # name of z1000 file
        z1000_file = 'atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_z1000_ll.nc',
        # name of tau input file
        tau_file = '/home/disk/rhodium/bowenliu/remap/atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-112_double_restart_100yearJanInit_tau300-700_ll.nc',
        # where to put output plots
        output_prefix = './scratch/2in-2out_TC-eval',
    )

if __name__ == "__main__":
    tc_main(**params_example)