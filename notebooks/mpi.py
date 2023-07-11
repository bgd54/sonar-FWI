from mpi4py import MPI
import sys
import os

sys.path.insert(0, os.path.abspath("/home/hajta2/study/sonar-FWI/cli"))
from simulation.sonar import Sonar
from simulation.utils import (
    CircleBottom,
    EllipsisBottom,
    run_beam,
    positions_line,
    gen_velocity_profile,
)
from simulation.sources import GaborSource
from simulation.plotting import plot_velocity

from examples.seismic import Receiver

import matplotlib.pyplot as plt
import numpy as np

from devito import configuration

configuration["mpi"] = True
configuration["language"] = "openmp"

domain_size = (6, 3)
radius = 2.8
v_env = 1.5
source_distance = 0.002
ns = 128
cy = (ns - 1) / 2 * source_distance
f0 = 50
spatial_dist = round(v_env / f0 / 3, 3) / 2
domain_dims = (
    round(domain_size[0] / spatial_dist),
    round(domain_size[1] / spatial_dist),
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    vp = gen_velocity_profile(
        EllipsisBottom(True), domain_dims, spatial_dist, v_water=v_env
    )
vp = comm.bcast(vp, root=0)
print(vp.shape)
sonar = Sonar(
    domain_size,
    f0,
    v_env,
    vp,
    source_distance=source_distance,
    ns=ns,
    spatial_dist=spatial_dist,
)
sonar.set_source()
sonar.finalize()

# sonar.op(time=sonar.time_range.num - 2, dt=sonar.model.critical_dt)
# plt.plot(sonar.rec.data[:, 64])
# plt.show()
