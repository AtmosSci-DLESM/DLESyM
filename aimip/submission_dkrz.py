import xarray as xr
import s3fs

fs_ice = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://s3.eu-dkrz-1.dkrz.cloud'},
    key = 'PvUwMS8lvawlRecf5Bnp',
    secret = 'NXrq2PGaBy2pig2Lhgmr85Bdk3tZtkYTsmsunrDh',
)

# 2) Define your local folder and target S3 prefix
local_dir = "/home/disk/mercury3/nacc/aimip_subission/university_of_washington/DLESyM"  # path on your computer
bucket = "ai-mip"          
remote_prefix = f"{bucket}/DLESyM/"   # where it will land in the bucket

print(f"Submitting {local_dir} to {remote_prefix}")
fs_ice.put(local_dir, remote_prefix, recursive=True)
print(f"Submission complete")