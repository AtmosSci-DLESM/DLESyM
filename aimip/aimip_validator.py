import xarray as xr
import os
import subprocess
import logging
import sys
import re

logger = logging.getLogger(__name__)

class AIMIPValidator:
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.ds = xr.open_dataset(self.filepath)
        self.fname = os.path.basename(self.filepath)
        self.f_parts = self.fname.replace('.nc', '').split('_')
        self.var_name = self.f_parts[0]

        self.units_library = {
            'ta': 'K',
            'tas': 'K',
            'zg': 'm',
            'z': 'm',
            'plev': 'Pa',
            'pressure': 'Pa',
        }

    def check_directory_consistency(self):
        path_parts = self.filepath.split(os.sep)
        # {inst}/{model}/{exp}/{member}/{freq}/{var}/{grid}/{version}
        mapping = {-3: 5, -4: 0, -5: 1, -6: 4, -7: 3, -8: 2}
        for depth, f_idx in mapping.items():
            if path_parts[depth] != self.f_parts[f_idx]:
                return False, f"Path part '{path_parts[depth]}' != Filename part '{self.f_parts[f_idx]}'"
        return True, "Consistent"

    # check that filename matechs variable name
    def check_filename_matches_variable(self):
        if self.f_parts[0] != self.var_name:
            return False, f"Filename part '{self.f_parts[0]}' != Variable name '{self.var_name}'"
        return True, "Consistent"

    # check that units are consistent
    def check_unit_consistency(self):
        if self.ds[self.var_name].attrs.get('units') != self.units_library[self.var_name]:
            return False, f"Units attribute '{self.ds[self.var_name].attrs.get('units')}' != '{self.units}'"
        return True, "Consistent"

    def check_pressure_units(self):
        if 'plev' in self.ds.coords:
            p_vals = self.ds['plev'].values
            if p_vals.max() < 2000:
                return False, f"Pressure values ({p_vals.max()}) likely hPa; CMIP requires Pa."
            if self.ds['plev'].attrs.get('units') != 'Pa':
                return False, "Units attribute not set to 'Pa'."
        return True, "Correct"
