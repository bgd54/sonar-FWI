#!/bin/sh
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH -o out/jupyter_a100.out
#SBATCH -e err/jupyter_a100.err

source /home/hajta3/.bashrc 
module load nvhpc/23.1-mpi-centos8
module avail
module list
hostname
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

cd /home/hajta3/sonar/sonar-FWI/notebooks
export DEVITO_LANGUAGE=openacc
export DEVITO_LOGGING=DEBUG
export DEVITO_PLATFORM=nvidiaX
export DEVITO_ARCH=nvc
conda activate devito

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.password=''
