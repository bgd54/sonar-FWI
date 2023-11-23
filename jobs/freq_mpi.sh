#!/bin/sh
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH -o out/gpu-%A  # send stdout to outfile
#SBATCH -e err/err-gpu-%A # send stderr to errfile

source /home/hajta3/.bashrc
module load nvhpc/23.1-mpi-centos8

cd /home/hajta3/sonar/sonar-FWI/cli
export DEVITO_LANGUAGE=openacc
export DEVITO_LOGGING=DEBUG
export DEVITO_PLATFORM=nvidiaX
export DEVITO_ARCH=nvc
export DEVITO_MPI=1
conda activate devito

mpirun -np 2 python -m simulation run-single-freq-ellipse 45 60 75 90 105 120 135 -x 60 -y 30 --f0 100 --v_env 1.5 --ns 128 --sd 0.002 --dir "/home/hajta3/sonar/sonar-FWI/jobs/sim_output" --plot 
