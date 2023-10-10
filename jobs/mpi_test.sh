#!/bin/sh
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=gpu
#SBATCH -o out/mpi-%A
#SBATCH -o err/mpi-%A

source /home/hajta2/.bashrc

conda activate devito
module load nvhpc/23.1-mpi
export OMPI_CC=ncv
export OMPI_CXX=nvc++
export DEVITO_LANGUAGE=openacc
export DEVITO_LOGGING=DEBUG
export DEVITO_JIT_BACKDOOR=0
export DEVITO_PLATFORM=nvidiaX
export DEVITO_ARCH=nvc
export DEVITO_AUTOTUNING=aggressive
export DEVITO_MPI=1

mpirun -np 2 python test-mpi.py
