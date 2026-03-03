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

    def run_cf_checker(self):

            cmd = [sys.executable, "-m", "cfchecker.cfchecks", "-v", "1.8", self.filepath]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                full_log = (result.stdout or "") + (result.stderr or "")

                # 1. Look for actual error lines (e.g., "ERROR: (3.1): ...")
                # This regex matches "ERROR" or "FATAL" at the start of a line, 
                # but ignores the summary "ERRORS detected: 0"
                error_pattern = re.compile(r'^(ERROR|FATAL):', re.MULTILINE)
                error_matches = error_pattern.findall(full_log)

                if error_matches:
                    # Extract the specific lines that failed for the report
                    error_lines = [line for line in full_log.split('\n') 
                                if line.startswith('ERROR:') or line.startswith('FATAL:')]
                    return False, "\n".join(error_lines)

                # 2. Confirm the check actually reached the end
                if "ERRORS detected:" in full_log:
                    # We found the summary and no specific ERROR: lines were matched
                    return True, "Passed"

                # 3. Handle cases where the checker crashed silently
                return False, f"CF-Checker aborted. No summary found. Output:\n{full_log[:300]}"
                
            except Exception as e:
                return False, f"Subprocess Error: {str(e)}"
            