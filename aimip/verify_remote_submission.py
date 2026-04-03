#!/usr/bin/env python3
"""
Verify that the remote DKRZ S3 submission was uploaded correctly.

Downloads the submission from S3 to a temporary directory and runs the full
AIMIP test suite (test_submission.py) against the downloaded files.
"""

import os
import sys
import tempfile
import subprocess
import shutil

import s3fs


def main():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud"},
        key=os.environ.get("DKRZ_S3_KEY", "PvUwMS8lvawlRecf5Bnp"),
        secret=os.environ.get("DKRZ_S3_SECRET", "NXrq2PGaBy2pig2Lhgmr85Bdk3tZtkYTsmsunrDh"),
    )

    bucket = "ai-mip"
    remote_prefix = f"{bucket}/DLESyM/"

    print(f"Downloading remote submission from s3://{remote_prefix}")
    tmpdir = tempfile.mkdtemp(prefix="aimip_remote_verify_")

    try:
        fs.get(remote_prefix, tmpdir, recursive=True)
        # Check where .nc files landed
        nc_files = []
        for root, _, files in os.walk(tmpdir):
            nc_files.extend(os.path.join(root, f) for f in files if f.endswith(".nc"))

        if not nc_files:
            print("ERROR: No .nc files found in downloaded content.")
            print(f"Downloaded to: {tmpdir}")
            for r, d, f in os.walk(tmpdir):
                print(f"  {r}: dirs={d}, files={f}")
            sys.exit(1)

        # Use the root of the downloaded tree (tmpdir) as the submission root
        # so glob **/*.nc finds files regardless of intermediate dirs
        submission_root = tmpdir
        print(f"Found {len(nc_files)} NetCDF file(s). Running test suite...")

        env = os.environ.copy()
        env["AIMIP_SUBMISSION_DIR"] = submission_root

        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(script_dir, "test_submission.py")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v"],
            env=env,
            cwd=script_dir,
        )

        sys.exit(result.returncode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"Cleaned up temporary directory {tmpdir}")


if __name__ == "__main__":
    main()
