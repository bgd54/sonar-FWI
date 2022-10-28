"""This module provides the sonar class for the water tank project."""
import numpy as np
import numpy.typing as npt
from devito import Eq, Operator, TimeFunction, solve
from examples.seismic import Model

from simulation import plotting, utils


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
        bottom: utils.Bottom,
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
        self.f0 = f0
        self.v_env = v_env
        exp = utils.find_exp(self.v_env / self.f0)
        self.size_x = (int)(size_x / 10**exp)
        self.size_y = (int)(size_y / 10**exp)
        shape = (self.size_x, self.size_y)
        spacing = (10**exp, 10**exp)
        origin = (0.0, 0.0)
        self.model = Model(
            vp=self.set_bottom(bottom,
                               cx=posx,
                               cy=posy if posy != 0.0 else
                               ((ns - 1) / 2 * v_env / f0 / 8) / size_y),
            origin=origin,
            spacing=spacing,
            shape=shape,
            space_order=2,
            nbl=10,
            bcs="damp",
        )
        print(
            f'spacing: {spacing}, size: {self.size_x} x {self.size_y}, dt: {self.model.critical_dt} t: {tn}'
        )
        (
            self.src,
            self.rec,
            self.time_range,
            self.center_pos,
            self.sdist,
        ) = utils.setup_domain(
            self.model,
            tn=tn,
            ns=ns,
            f0=self.f0,
            posx=posx,
            posy=posy,
            v_env=self.v_env,
        )
        self.u = TimeFunction(name="u",
                              grid=self.model.grid,
                              time_order=2,
                              space_order=2)
        pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        stencil = Eq(self.u.forward, solve(pde, self.u.forward))
        src_term = self.src.inject(
            field=self.u.forward,
            expr=self.src * self.model.critical_dt**2 / self.model.m,
        )
        rec_term = self.rec.interpolate(expr=self.u)

        self.op = Operator([stencil] + src_term + rec_term,
                           subs=self.model.spacing_map,
                           openmp=True)

    def set_bottom(self,
                   bottom: utils.Bottom,
                   cx: float,
                   cy: float,
                   v_wall: float = 3.24) -> npt.NDArray:
        """Set the bottom of the water tank.

        Args:
            bottom (Enum): Bottom of the water tank.
        """
        v = np.full((self.size_x, self.size_y), self.v_env, dtype=np.float32)
        if bottom == utils.Bottom.ellipsis:
            nx = self.size_x
            nz = self.size_y
            a = (int)((nx - 1) / 2)
            b = (int)((nz - 1) / 2)
            for i in range(nx):
                for j in range(nz):
                    if ((i - a)**2 / a**2 + (j - b)**2 / b**2) > 1:
                        v[i, j] = v_wall
            v[:, :b] = self.v_env
        elif bottom == utils.Bottom.flat:
            y_wall = max(int(self.size_y * 0.8), self.size_y - 50)
            v[:, y_wall:] = v_wall
        elif bottom == utils.Bottom.circle:
            ox = int(cx * self.size_x)
            oy = int(cy * self.size_y)
            r = self.size_y - oy - 10
            x = np.arange(0, v.shape[0])
            y = np.arange(0, v.shape[1])
            mask = (y[np.newaxis, :] - oy)**2 + (x[:, np.newaxis] -
                                                 ox)**2 < r**2
            v[mask] = v_wall
        return v

    def run_position_angles(
        self,
        angle_left: float,
        angle_right: float,
        angle_step: float,
    ) -> None:
        """Run the sonar simulation. Plots the results.

        Args:
            angle_left (float): Left angle of the simulation. (deg)
            angle_right (float): Right angle of the simulation. (deg)
            angle_step (float): Angle step of the simulation. (deg)
        """
        angles = np.arange(angle_left, angle_right, angle_step)
        results = utils.run_positions_angles(
            self.model,
            self.src,
            self.rec,
            self.op,
            self.u,
            self.v_env,
            self.sdist,
            self.time_range,
            centers=[self.center_pos],
            angle=angles,
        )

        res2 = utils.calculate_coordinates(
            self.model.domain_size,
            rec_pos=[self.center_pos],
            angle=angles,
            distance=results[0],
            amplitude=results[1],
        )

        plotting.compare_velocity_to_measure(
            self.model,
            res2[0],
            source=self.src.coordinates.data,
            receiver=self.rec.coordinates.data,
        )

    def plot_model(self, plot: plotting.PlotType) -> None:
        """Plot the model."""
        print(self.rec.data.shape)
        if plot == plotting.PlotType.model:
            plotting.plot_velocity(self.model, self.src.coordinates.data,
                                   self.rec.coordinates.data)

    def run_angles(self, angles: npt.NDArray) -> npt.NDArray:
        """Run the sonar simulation. Plots the results.

        Args:
            angles (NDArray[float]): list of angles to launch beams 
        """
        return utils.run_angles(
            self.model,
            self.src,
            self.rec,
            self.op,
            self.u,
            self.sdist,
            self.time_range,
            center=self.center_pos,
            angle=angles,
        )

    def parse_and_plot(self, angles, recordings):
        distances = np.zeros(angles.shape)
        for i, (alpha, rec) in enumerate(zip(angles, recordings)):
            distances[i], _ = utils.object_distance(np.average(rec, axis=1),
                                                    self.model.critical_dt,
                                                    self.v_env)

        abs_coords = utils.calculate_coordinates_from_pos(
            rec_pos=self.center_pos, angle=angles, distance=distances)

        plotting.compare_velocity_to_measure(
            self.model,
            abs_coords,
            source=self.src.coordinates.data,
            receiver=self.rec.coordinates.data)
