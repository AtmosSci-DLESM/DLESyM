import xarray as xr
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class AIMIPValidator:
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.ds = xr.open_dataset(self.filepath)
        self.fname = os.path.basename(self.filepath)
        self.f_parts = self.fname.replace('.nc', '').split('_')
        self.var_name = self.f_parts[0]

    def check_directory_consistency(self):
        path_parts = self.filepath.split(os.sep)
        # {inst}/{model}/{exp}/{member}/{freq}/{var}/{grid}/{version}
        mapping = {-3: 5, -4: 0, -5: 1, -6: 4, -7: 3, -8: 2}
        for depth, f_idx in mapping.items():
            if path_parts[depth] != self.f_parts[f_idx]:
                return False, f"Path part '{path_parts[depth]}' != Filename part '{self.f_parts[f_idx]}'"
        return True, "Consistent"

    def check_pressure_units(self):
        if 'pressure' in self.ds.coords:
            p_vals = self.ds['pressure'].values
            if p_vals.max() < 2000:
                return False, f"Pressure values ({p_vals.max()}) likely hPa; CMIP requires Pa."
            if self.ds['pressure'].attrs.get('units') != 'Pa':
                return False, "Units attribute not set to 'Pa'."
        return True, "Correct"

    def run_cf_checker(self):
        try:
            result = subprocess.run(['cf-checker', '-v', '1.8', self.filepath], 
                                     capture_output=True, text=True)
            if "ERROR" in result.stdout or "FATAL" in result.stdout:
                return False, result.stdout
            return True, "Passed"
        except FileNotFoundError:
            return None, "cf-checker not installed"