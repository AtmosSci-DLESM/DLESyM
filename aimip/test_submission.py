import pytest
from aimip_validator import AIMIPValidator
import glob
import os

# Discover all netcdf files in your submission directory
output_files = glob.glob('/home/disk/mercury3/nacc/aimip_subission/**/*.nc', recursive=True)

@pytest.mark.parametrize("filepath", output_files)
class TestAIMIPSubmission:

    def test_file_naming_convention(self, filepath):
        fname = os.path.basename(filepath)
        assert len(fname.split('_')) == 7, f"Invalid filename structure: {fname}"

    def test_directory_hierarchy(self, filepath):
        validator = AIMIPValidator(filepath)
        success, msg = validator.check_directory_consistency()
        assert success, msg

    def test_data_precision(self, filepath):
        validator = AIMIPValidator(filepath)
        assert validator.ds[validator.var_name].dtype == 'float32'

    def test_vertical_coords(self, filepath):
        validator = AIMIPValidator(filepath)
        success, msg = validator.check_pressure_units()
        assert success, msg

    def test_cf_compliance(self, filepath):
        validator = AIMIPValidator(filepath)
        status, msg = validator.run_cf_checker()
        if status is None:
            pytest.skip("cf-checker utility not found")
        assert status, f"CF Compliance Failed: {msg}"