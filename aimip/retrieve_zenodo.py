import requests
import logging
import os
from tqdm import tqdm # Optional: for a nice progress bar

logger = logging.getLogger(__name__)

def download_zenodo_record(record_id, output_directory):
    # 1. Get metadata from Zenodo API
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    logger.info(f"Downloading Zenodo record {record_id}...")

    # create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 2. Iterate through files (this record typically has one main .nc file)
    for file_info in data['files']:
        download_url = file_info['links']['self']
        filename = file_info['key']
        file_size = file_info['size']

        logger.info(f"Downloading {filename} ({file_size / 1e6:.2f} MB)...")
        
        # 3. Stream the download
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(output_directory, filename), 'wb') as f:
                # Using tqdm for a progress bar
                for chunk in tqdm(r.iter_content(chunk_size=8192), 
                                  total=file_size//8192, unit='KB'):
                    f.write(chunk)
        logger.info(f"Finished: {filename}")

# Run for your specific ID
download_zenodo_record("17065758", "forcing_data/")