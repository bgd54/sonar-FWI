import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


sys.path.insert(0, os.path.abspath("/home/hajta2/sonar/sonar-FWI/cli"))
from simulation.sources import GaborSource
from simulation.sonar import Sonar
from simulation.utils import EllipsisBottom, gen_velocity_profile


domain_size = (60, 30)
v_env = 1.5
ns = 128
source_distance = 0.002
f0 = 50
space_order = 8
spatial_dist = round(v_env / f0 / 3, 3)
dt = spatial_dist / 20

domain_dims = (
    round(domain_size[0] / spatial_dist),
    round(domain_size[1] / spatial_dist),
)
vp = gen_velocity_profile(EllipsisBottom(True), domain_dims, spatial_dist, v_water = v_env)


from examples.seismic import Model
model = Model(
    vp=vp,
    origin=(0.0, 0.0),
    spacing=(spatial_dist, spatial_dist),
    shape=domain_dims,
    space_order=space_order,
    nbl=(ns - 1) / 2 * source_distance / dt,
    bcs="damp",
    dt=dt,
    dtype=np.float64,
)
