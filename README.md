<p align="center">
  <img src="storm.gif" alt="Storm animation" width="850">
</p>


# DLESyM
Code repository for training, running and analyzing a Deep Learning Earth System Model (DLESyM) as presented our paper ["A Deep Learning Earth System Model for Efficient Simulation of the Observed Climate"](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV001706) (arXiv manuscript [here](https://arxiv.org/abs/2409.16247)). This repository is accompanied by a data store (linked [here](https://dlesym.atmos.washington.edu/DLESyM_AGU-Advances/)) hosted at the University of Washington that provides free access to training data, simulation output, and cached analysis output. Model checkpoints are included in this repository, but require GitLFS. 

Together, this repository and the associated data store are sufficient to reproduce results presented in Cresswell-Clay et al. 2024. If you would like to start running DLESyM as quickly as possible without necessarily reproducing all of the results from the study, you will only need to complete the next two sections: "Setting up your environment" and "Inference with DLESyM". 

For any feedback on this repo, or suggestions for your use case, please feel free to reach out to me directly.


## Setting up your environment

Before using DLESyM, you'll need to set up the repository and environemnt on your machine. Your machine must have Git and GitLFS installed (instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [here](https://git-lfs.github.com/), repectively).

1.  Clone the repository:
    ```sh
    git clone https://github.com/AtmosSci-DLESM/DLESyM.git
    ```

2. Ensure LFS files are included (model checkpoints, initialization data example)
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

**Whenever running inference, trianing or evaluations, you need this environment to be activated**


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

## Experiment Replication 

## Analysis Replication 

This repo is designed for exact replication of the results presented in [Cresswell-Clay et al. 2024](https://arxiv.org/abs/2409.16247). Code for analysis routines, as well as instructions for their use are provided in the `DLESyM/evaluation/ `. 