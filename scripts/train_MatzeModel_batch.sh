#!/bin/bash
##################################################################
# REQUIRED DIRECTIVES   ------------------------------------------
##################################################################
# Account to be charged
##SBATCH --account=ONRDC51005684
##SBATCH -q debug

# Select nodes
#SBATCH --nodes=1

# Total tasks count
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpu_bind=sockets

# Set max wall time
#SBATCH --time=00:05:00

# Request GPU nodes – select one of the options below
#SBATCH --gres=gpu:A100:2 # Same as mla node
##SBATCH --constraint=mla

##################################################################
# OPTIONAL DIRECTIVES  -------------------------------------------
##################################################################
# Name the job
# SBATCH --jobname=train_MatzeModel

# Change stdout and stderr filenames
#SBATCH --output=/home/disk/quicksilver/nacc/dlesm/zephyr/scripts/train_MatzeModel.out
#SBATCH --error=/home/disk/quicksilver/nacc/dlesm/zephyr/scripts/train_MatzeModel.out

##################################################################
# EXECUTION BLOCK ------------------------------------------------
##################################################################

#################################################################
# Job information set by Baseline Configuration variables
#################################################################
echo ----------------------------------------------------------
echo "Type of node                    " $BC_NODE_TYPE
echo "CPU cores per node              " $BC_CORES_PER_NODE
echo "CPU cores per standard node     " $BC_STANDARD_NODE_CORES
echo "CPU cores per accelerator node  " $BC_ACCELERATOR_NODE_CORES
echo "CPU cores per big memory node   " $BC_BIGMEM_NODE_CORES
echo "Hostname                        " $BC_HOST
echo "Maxumum memory per nodes        " $BC_MEM_PER_NODE
echo "Number of tasks allocated       " $BC_MPI_TASKS_ALLOC
echo "Number of nodes allocated       " $BC_NODE_ALLOC
echo "Working directory               " $WORKDIR
echo ----------------------------------------------------------

##############################################################
# Output some useful job information.  
##############################################################
echo "-------------------------------------------------------"
echo "Project ID                      " $SLURM_JOB_ACCOUNT
echo "Job submission directory        " $SLURM_SUBMIT_DIR
echo "Submit host                     " $SLURM_SUBMIT_HOST
echo "Job name                        " $SLURM_JOB_NAME
echo "Job identifier (SLURM_JOB_ID)   " $SLURM_JOB_ID
echo "Job identifier (SLURM_JOBID)    " $SLURM_JOBID
echo "Working directory               " $WORKDIR
echo "Job partition                   " $SLURM_JOB_PARTITION
echo "Job queue (QOS)                 " $SLURM_JOB_QOS
echo "Job number of nodes             " $SLURM_JOB_NUM_NODES
echo "Job node list                   " $SLURM_JOB_NODELIST
echo "Number of nodes                 " $SLURM_NNODES
echo "Number of tasks                 " $SLURM_NTASKS
echo "Node list                       " $SLURM_NODELIST
echo "-------------------------------------------------------"
echo

# move to zephyr directory and set environment variables for training 
cd /home/disk/quicksilver/nacc/dlesm/zephyr
source /home/disk/brume/nacc/.bashrc
source activate zephyr-1.0
export HYDRA_FULL_ERROR=1
export HDF5_USE_FILE_LOCKING=False
##export CUDA_VISIBLE_DEVICES = "2,3"
PYTHONPATH=/home/disk/quicksilver/nacc/dlesm/zephyr

./scripts/train_MatzeModel.sh

exit
