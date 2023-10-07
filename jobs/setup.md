# How To Set Up Environment on ITK HPC Cluster

## Install Miniconda3

```bash
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
```

## Create Conda Environment

```bash
conda create --name devito python=3.11
conda activate devito
```

## Install Devito and Dependencies

```bash
module load nvhpc/23.1-mpi-centos8
export OMPI_CC=gcc
export OMPI_CXX=g++
git clone git@github.com:devitocodes/devito.git
cd devito
pip3 install -r requirements.txt
pip3 install -r requirements-mpi.txt
pip3 install -r requirements-testing.txt
pip3 install devito
```

## Run an Example

```bash
DEVITO_MPI=1 DEVITO_LANGUAGE=openacc DEVITO_LOGGING=DEBUG DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=nvc mpirun -np 2 python3 ./examples/seismic/acoustic/acoustic_example.py
```
