"""This module provides the sonar class for the water tank project."""
import numpy as np

from simulation import utils, plotting
from devito import Eq, Operator, TimeFunction, solve
from examples.seismic import Model


class Sonar:
    """Sonar class for the water tank project."""

    def __init__(
        self,
        size_x: int,
        size_y: int,
        f0: float,
        v_env: float,
        tn: float,
        ns: int,
        posx: float,
        posy: float,
    ) -> None:
        """Initialize the sonar class.

        Args:
            size_x (int): Size in x direction. (m)
            size_y (int): Size in y direction. (m)
            f0 (float): Center frequency of the signal. (kHz)
            v_env (float): Environment velocity. (km/s)
            tn (float): End time of the simulation. (ms)
            ns (int): Number of sources.
            posx (float): Position of the source in x direction. (relative)
            posy (float): Position of the source in y direction. (relative)
        """
        self.size_x = size_x
        self.size_y = size_y
        self.f0 = f0
        self.v_env = v_env
        exp = utils.find_exp(self.v_env / self.f0)
        spacing = (10**exp, 10**exp)
        origin = (0.0, 0.0)
        v = np.full((self.size_x, self.size_y), self.v_env, dtype=np.float32)
        self.model = Model(
            origin=origin,
            spacing=spacing,
            shape=(size_x, size_y),
            space_order=2,
            nbl=10,
            bcs="damp",
            vp=v,
        )
        (
            self.src,
            self.rec,
            self.time_range,
            self.center_pos,
            self.sdist,
        ) = utils.setup_domain(
            self.model, tn=tn, ns=ns, f0=self.f0, posx=posx, posy=posy, v_env=self.v_env
        )
        self.u = TimeFunction(
            name="u", grid=self.model.grid, time_order=2, space_order=2
        )
        pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        stencil = Eq(self.u.forward, solve(pde, self.u.forward))
        src_term = self.src.inject(
            field=self.u.forward,
            expr=self.src * self.model.critical_dt**2 / self.model.m,
        )
        rec_term = self.rec.interpolate(expr=self.u)

        self.op = Operator(
            [stencil] + src_term + rec_term, subs=self.model.spacing_map, openmp=True
        )

    def run_position_angles(
        self,
        angle_left: float,
        angle_right: float,
        angle_step: float,
    ) -> None:
        """Run the sonar simulation. Plots the results.

        Args:
            tn (float): End time of the simulation. (s)
            ns (int): Number of sources.
            angle_left (float): Left angle of the simulation. (deg)
            angle_right (float): Right angle of the simulation. (deg)
            angle_step (float): Angle step of the simulation. (deg)
        """
        angles = np.arange(angle_left, angle_right, angle_step)
        results = utils.run_position_angle(
            self.model,
            self.src,
            self.rec,
            self.op,
            self.u,
            self.v_env,
            self.sdist,
            self.time_range,
            self.posx,
            self.posy,
            angles,
        )

        res2 = utils.calculate_coordinates(
            self.model.domain_size,
            rec_pos=[self.center_pos],
            sdist=self.sdist,
            angle=angles,
            distance=results[0],
            amplitude=results[1],
        )

        plotting.compare_velocity_to_measure(
            self.model,
            res2[0],
            source=self.src.coordinates.data[0],
            receiver=self.rec.coordinates.data[0],
        )
