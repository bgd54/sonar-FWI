"""This module provides the sonar class for the water tank project."""
import numpy as np

from sonar import utils, plotting
from devito import Eq, Operator, TimeFunction, solve
from examples.seismic import Model


class Sonar:
    """Sonar class for the water tank project."""

    def __init__(self, size_x: int, size_y: int, f0: float, v_env: float):
        """Initialize the sonar class.

        Args:
            size_x (int): Size in x direction. (m)
            size_y (int): Size in y direction. (m)
            f0 (float): Center frequency of the signal. (kHz)
            v_env (float): Environment velocity. (km/s)
        """
        self.size_x = size_x
        self.size_y = size_y
        self.f0 = f0
        self.v_env = v_env
        exp = utils.find_exp(self.v_env / self.f0)
        spacing = (10 ** exp, 10 ** exp)
        origin = (0.0, 0.0)
        v = np.full((self.size_x, self.size_y), self.v_env, dtype=np.float32)
        self.model = Model(origin=origin, spacing=spacing, shape=(size_x, size_y),
                           space_order=2, nbl=10, bcs="damp", vp=v)

    def run(self, tn: float, ns: int, posx: float, posy: float, angle: float):
        """Run the sonar simulation.

        Args:
            tn (float): End time of the simulation. (s)
            ns (int): Number of sources.
            posx (float): Position of the source in x direction. (m)
            posy (float): Position of the source in y direction. (m)
            angle (float): Angle of the source.
        """
        src, rec, time_range, center_pos, sdist = utils.setup_domain(
            self.model, tn=tn, ns=ns, f0=self.f0,
            posx=posx, posy=posy, v_env=self.v_env)
        u = TimeFunction(name="u", grid=self.model.grid, time_order=2,
                         space_order=2)
        pde = self.model.m * u.dt2 - u.laplace + self.model.damp * u.dt
        stencil = Eq(u.forward, solve(pde, u.forward))
        src_term = src.inject(field=u.forward,
                              expr=src * self.model.critical_dt**2 / self.model.m)
        rec_term = rec.interpolate(expr=u)

        op = Operator([stencil] + src_term + rec_term,
                      subs=self.model.spacing_map,
                      openmp=True)
        results = utils.run_position_angle(self.model, src, rec, op, u,
                                           self.v_env, sdist, time_range,
                                           posx, posy, angle)

        res2 = utils.calculate_coordinates(self.model.domain_size, rec_pos=[center_pos],
                                           sdist=sdist, angle=angle, distance=results[0],
                                           amplitude=results[1])
        
        plotting.compare_velocity_to_measure(self.model, res2[0], source=src.coordinates.data[0],
                                             receiver=rec.coordinates.data[0])
