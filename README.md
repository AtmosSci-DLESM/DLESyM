# DLESyM
Code repository for training, running and analyzing a Deep Learning Earth System Model (DLESyM) as presented in [Cresswell-Clay et al. 2024](https://arxiv.org/abs/2409.16247). Full use of this repository requires Git and Git LFS (instructions for downloading can be found [here](https://github.com/git-guides/install-git) and [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), repsectively). 

## Organization


```
   |-data_processing - utilities for data curation
   |-environments - conda environemnts files 
   |-evaluation
   |-example_data - contains sample initialization
   |-models - contains the model files for DLESyMls 
   |-scripts - inference, training utilities, example batch scripts
   |-testing 
   |-training
   |---configs - configurations used for model initialization
   |---dlwp - classes and utilities used for model interence,
```

## Getting Started
Before messing around with DLESyM, you'll need to set up the repository and environemnt on your machine. You'll need to make sure your machine has Git and Git LFS installed (instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [here](https://git-lfs.github.com/), repectively).

1.  Clone the repository:
    ```sh
    git clone https://github.com/AtmosSci-DLESM/DLESyM.git
    ```

2. Ensure LFS files are included
    ```sh
    git lfs pull
    ```

3. Create the virtual environment. Conda is recommended for environemnt management (it's what we used). Code in this repo was developed using `conda 4.13.0`; install instructions found [here](https://docs.anaconda.com/miniconda/). Once conda is installed, create the environment: 
    ```sh
    conda env create -f /path/to/DLESyM/environments/dlesym-0.1.yaml
    ```
    activate the environment: 
    ```sh
    conda activate dlesym-0.1
    ```

Now you're all set! 

NOTE: For modification of DLESyM source, I suggest creating a fork. When forking, make sure you clone the fork and not the original repo. Otherwise, the above steps are the same. 

## Inference with DLESyM

Once you've cloned the repo, created the environment, and pulled initialization data (Git LFS files), you're ready for inference. Analyses presented in the corresponding study were performed on a 100 year simulation initialized January 1st 2017. 

To recreate this simulation, follow these steps: 

1. Make sure the environment you've just created is active: 

    ```sh
    conda activate dlesym-0.1
    ```

2. `scripts/example_100yr_forecasts_12init.sh` provides an example call to the coupled forecast script that will reproduce the 100 year simulation. First, we need to adjust a couple of the default parameters. 
    - `MODULE_DIR` should be set to the absolute path of your `DLESyM` clone. 
    - `DEVICE_NUMBERS` should be `-1` if using CPU, or the index of your CUDA compatible GPU. - `OUTPUT_DIR` is the directory in which simulation output is saved. It needs to be large enough to store the full 100 years of output (about 250GB). 
    - `OUTPUT_FILE` is where logs, stdout, stderr will be piped. Commenting this line out will sent these directly to your terminal session. 
    - The remaining parameters should be fine as default values for now.

3. Time to run the simulation: 

    ```sh
    ./scripts/example_100yr_forecasts_12init.sh
    ```

