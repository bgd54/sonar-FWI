import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath('/home/hajta2/sonar/sonar-FWI/cli/'))
from simulation.sonar import Sonar
from simulation.utils import EllipsisBottom, gen_velocity_profile
from simulation.sources import GaborSource
from mpi4py import MPI

from devito import configuration
configuration['mpi'] = True

domain_size = (60, 30)
v_env = 1.5
ns = 128
source_distance = 0.002
f0 = 325
space_order = 8
spatial_dist = round(v_env / f0 / 3, 3)
dt = spatial_dist / 20
sonar = Sonar(domain_size, f0, v_env, EllipsisBottom(True), source_distance=source_distance, ns=ns, spatial_dist=spatial_dist)
sonar.set_source()
sonar.finalize()

src = sonar.src
alpha = 45
dt = sonar.model.critical_dt
ns = src.coordinates.data.shape[0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()
if alpha <= 90:
    max_latency = (
        np.cos(np.deg2rad(alpha)) * ((ns - 1) * source_distance / v_env) / dt
    )
elif alpha > 90:
    max_latency = np.cos(np.deg2rad(alpha)) * (source_distance / v_env) / dt

all_data = comm.gather(src.data, root=0)

if rank == 0:
    full_data = np.concatenate(all_data, axis=1)

    for i in range(full_data.shape[1]):
        latency = -np.cos(np.deg2rad(alpha)) * (i * source_distance / v_env)
        full_data[:, i] = np.roll(full_data[:, i], int(latency / dt + max_latency))

    divided_data = np.array_split(full_data, size, axis=1)
else:
    divided_data = None

new_data = comm.scatter(divided_data, root=0)
np.copyto(src.data, new_data)
print(f"Simulation took {time.time() - start_time} seconds")
sonar.u.data.fill(0)
sonar.op(time=sonar.time_range.num - 2, dt=dt)
