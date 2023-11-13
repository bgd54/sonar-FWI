import math
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from devito import configuration
from examples.seismic import Model, TimeAxis, WaveletSource

class GaborSource(WaveletSource):
    def __init_finalize__(self, *args, **kwargs):
        super(GaborSource, self).__init_finalize__(*args, **kwargs)

    @property
    def wavelet(self):
        assert self.f0 is not None
        agauss = 0.5 * self.f0
        tcut = self.t0 or 5 / agauss
        s = (self.time_values - tcut) * agauss
        a = self.a or 1
        return a * np.exp(-0.5 * s**2) * np.cos(2 * np.pi * s)

# set the mpi configuration
configuration["mpi"] = True
configuration["language"] = "C"

# Domain initialization
domain_size = (60, 30)
v_env = 1.5
ns = 128
source_distance = 0.002
f0 = 50
space_order = 8
spatial_dist = v_env / f0 / 3
dt = spatial_dist / 20

model = Model(
    vp=np.ones(domain_size) * v_env,
    origin=(0.0, 0.0),
    spacing=(spatial_dist, spatial_dist),
    shape=(
        round(domain_size[0] / spatial_dist),
        round(domain_size[1] / spatial_dist),
    ),
    space_order=space_order,
    nbl=(ns - 1) / 2 * source_distance / dt,
    bcs="damp",
    dt=dt,
    dtype=np.float64,
)

max_distance = math.sqrt((domain_size[0] / 2) ** 2 + domain_size[1] ** 2)
time_range = TimeAxis(start=0.0, stop=max_distance * 2 / v_env + 5, step=model.critical_dt)

