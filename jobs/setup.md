# How To Set Up Environment on ITK HPC Cluster

## Install Miniconda3

```bash
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
```

## Create Conda Environment

```bash
conda create --name devito python=3.11
conda activate devito
```

## Install Devito and Dependencies

```bash
module load nvhpc/23.1-mpi(-centos8)
export OMPI_CC=gcc
export OMPI_CXX=g++
git clone git@github.com:devitocodes/devito.git
cd devito
pip3 install -r requirements.txt
pip3 install -r requirements-mpi.txt
pip3 install -r requirements-testing.txt
pip3 install devito matplotlib tqdm typer
```

## Run an Example

```bash
export OMPI_CC=nvc
export OMPI_CXX=nvc++
DEVITO_MPI=1 DEVITO_LANGUAGE=openacc DEVITO_LOGGING=DEBUG DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=nvc mpirun -np 2 python3 ./examples/seismic/acoustic/acoustic_example.py
```

## Run Jupyter Notebook on Cluster

```bash
pip install ipcluster notebook
cd devito
./scripts/create_ipyparallel_mpi_profile.sh
srun --gres=gpu:a100:2 -n 2 --partition=gpu --pty bash -i
conda activate devito
ipcluster start -n 2 --profile mpi --engines mpi
```

Open up another terminal

```bash
ssh -L 8888:<ssh_host>:8888 <ssh_host>
conda activate devito
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.password='' --NotebookApp.allow_origin='*'
```

## Environment Variables

### GPU

```bash
export DEVITO_LANGUAGE=openacc
export DEVITO_LOGGING=DEBUG
export DEVITO_JIT_BACKDOOR=0
export DEVITO_PLATFORM=nvidiaX
export DEVITO_ARCH=nvc
```

### CPU

```bash
export DEVITO_LANGUAGE=openmp
export DEVITO_ARCH=gcc
export DEVITO_LOGGING=DEBUG
```

### MPI

```bash
export DEVITO_MPI=1
```
