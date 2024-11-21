# DLESyM
Code repository for training, running and analyzing a Deep Learning Earth System Model (DLESyM). Full use of this repository requires Git and Git LFS (instructions for downloading can be found [here](https://github.com/git-guides/install-git) and [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), repsectively). 

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
In order to access model files make sure Git LFS is installed (instructions [here](https://git-lfs.github.com/)). 