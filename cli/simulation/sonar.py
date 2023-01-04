"""This module provides the sonar class for the water tank project."""
import math
import time
import tqdm

import numpy as np
import numpy.typing as npt
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
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
        ns: int,
        posx: float,
        posy: float,
        bottom: utils.Bottom,
        source_distance: float,
        snapshot_delay: float = 0.0,
        obstacle: bool = False,
        r: float = 28.0,
    ) -> None:
        """Initialize the sonar class.

        Args:
            size_x (int): Size in x direction. (m)
            size_y (int): Size in y direction. (m)
            f0 (float): Center frequency of the signal. (kHz)
            v_env (float): Environment velocity. (km/s)
            ns (int): Number of sources.
            posx (float): Position of the source in x direction. (relative)
            posy (float): Position of the source in y direction. (relative)
            snapshot_delay (float): Time delay between snapshots (ms)
        """
        self.f0 = f0
        self.v_env = v_env
        self.sdist = source_distance
        self.spatial_dist = round(self.v_env / self.f0 / 3, 3)
        self.size_x = (int)(size_x / self.spatial_dist + 1)
        self.size_y = (int)(size_y / self.spatial_dist + 1)
        self.obstacle = obstacle
        self.real_dist = np.zeros(121)
        shape = (self.size_x, self.size_y)
        spacing = (self.spatial_dist, self.spatial_dist)
        origin = (0.0, 0.0)
        posy = posy if posy != 0.0 else ((ns - 1) / 2 * self.sdist) / size_y
        travel_distance = math.sqrt(
            (size_x * (posx if posx >= 0.5 else 1 - posx)) ** 2
            + (size_y * (posy if posy >= 0.5 else 1 - posy)) ** 2
        )
        self.tn = travel_distance * 2 / v_env + 5
        self.model = Model(
            vp=self.set_bottom(bottom, cx=posx, cy=posy, r=r),
            origin=origin,
            spacing=spacing,
            shape=shape,
            space_order=2,
            nbl=10,
            bcs="damp",
        )
        (self.src, self.rec, self.time_range, self.center_pos,) = utils.setup_domain(
            self.model,
            tn=self.tn,
            ns=ns,
            f0=self.f0,
            posx=posx,
            posy=posy,
            sdist=self.sdist,
            v_env=self.v_env,
        )
        print(
            f"spacing: {spacing}, size: {self.size_x} x {self.size_y}, {self.model.domain_size}\n"
            f"dt: {self.model.critical_dt} t: {self.tn}\n"
            f"rec_pos {{{posx}, {posy}}} -> {{{posx*size_x}, {posy * size_y}}} cp: {self.center_pos}, sdist = {self.sdist}"
        )
        if bottom == utils.Bottom.circle:
            print(f"{bottom} {r}")
        else:
            print(f"{bottom} o: {self.obstacle}")
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

        save_stencil = []
        if snapshot_delay > 0.0:
            snapshot_delay_iter = math.ceil(snapshot_delay / self.model.critical_dt)
            nsnaps = math.ceil(self.time_range.num / snapshot_delay_iter)
            time_subsampled = ConditionalDimension(
                "t_sub", parent=self.model.grid.time_dim, factor=snapshot_delay_iter
            )
            self.usave = TimeFunction(
                name="usave",
                grid=self.model.grid,
                time_order=2,
                space_order=2,
                save=nsnaps,
                time_dim=time_subsampled,
            )
            save_stencil.append(Eq(self.usave, self.u))

        self.op = Operator(
            [stencil] + save_stencil + src_term + rec_term,
            subs=self.model.spacing_map,
            openmp=True,
        )

    def set_bottom(
        self,
        bottom: utils.Bottom,
        cx: float,
        cy: float,
        v_wall: float = 3.24,
        r: float = None,
    ) -> npt.NDArray:
        """Set the bottom of the water tank.

        Args:
            bottom (Enum): Bottom of the water tank.
        """
        v = np.full((self.size_x, self.size_y), self.v_env, dtype=np.float32)
        if bottom == utils.Bottom.ellipsis:
            nx = self.size_x
            nz = self.size_y
            offs = round((1 / self.spatial_dist) / 2)
            a = round((nx - (1 / self.spatial_dist)) / 2)
            b = round((nz - (1 / self.spatial_dist)) / 2)
            x = np.arange(0, v.shape[0])
            y = np.arange(0, v.shape[1])
            mask = (y[np.newaxis, :] - offs - b) ** 2 / b**2 + (
                x[:, np.newaxis] - offs - a
            ) ** 2 / a**2 > 1
            v[mask] = v_wall
            if self.obstacle:
                r = v.shape[0] / 100
                ox = np.arange(offs, 2 * a + offs + 1, 2 * a / 50)
                oy = np.sqrt(1 - (ox - a - offs) ** 2 / a**2) * b + offs + b
                for i in range(0, 61):
                    self.real_dist[i] = 33.5 - i * (5 / 60)
                    self.real_dist[120 - i - 1] = self.real_dist[i]
                x = np.arange(0, v.shape[0])
                y = np.arange(0, v.shape[1])
                for oxx, oyy in tqdm.tqdm(zip(ox, oy)):
                    mask = (y[np.newaxis, :] - oyy) ** 2 + (
                        x[:, np.newaxis] - oxx
                    ) ** 2 < r**2
                    v[mask] = v_wall
            v[offs:-offs, :b] = self.v_env
        elif bottom == utils.Bottom.flat:
            y_wall = max(
                int(self.size_y * 0.8), round(self.size_y - 5 / self.spatial_dist)
            )
            v[:, y_wall:] = v_wall
            if self.obstacle:
                r = v.shape[0] / 100
                for i in tqdm.tqdm(range(1, 101, 2)):
                    a, b = v.shape[0] / 100 * i, y_wall
                    y, x = np.ogrid[-a : v.shape[0] - a, -b : v.shape[1] - b]
                    v[x * x + y * y <= r * r] = v_wall
        elif bottom == utils.Bottom.circle:
            ox = int(cx * self.size_x)
            oy = int(cy * self.size_y)
            r = round(r / self.spatial_dist)
            x = np.arange(0, v.shape[0])
            y = np.arange(0, v.shape[1])
            mask = (y[np.newaxis, :] - oy) ** 2 + (x[:, np.newaxis] - ox) ** 2 > r**2
            v[mask] = v_wall
            self.real_dist = np.full(121, 27)
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
        if plot == plotting.PlotType.model:
            plotting.plot_velocity(
                self.model, self.src.coordinates.data, self.rec.coordinates.data
            )

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
        for i, (alpha, rec) in tqdm.tqdm(enumerate(zip(angles, recordings))):
            distances[i], _ = utils.echo_distance(
                np.average(rec, axis=1),
                self.model.critical_dt,
                self.src.signal_packet,
                self.v_env,
                cut_ms=10.0,
            )

        mse = np.square(np.subtract(self.real_dist, distances)).mean()

        abs_coords = utils.calculate_coordinates_from_pos(
            rec_pos=self.center_pos, angle=angles, distance=distances
        )

        plotting.compare_velocity_to_measure(
            self.model,
            abs_coords,
            source=self.src.coordinates.data,
            receiver=self.rec.coordinates.data,
        )

        return mse

    def parse_and_plot_prom(self, angles, recordings):
        distances = np.zeros(angles.shape)
        for i, (alpha, rec) in tqdm.tqdm(enumerate(zip(angles, recordings))):
            distances[i], _ = utils.object_distance(
                np.average(rec, axis=1), self.model.critical_dt, self.v_env
            )

        mse = np.square(np.subtract(self.real_dist, distances)).mean()

        abs_coords = utils.calculate_coordinates_from_pos(
            rec_pos=self.center_pos, angle=angles, distance=distances
        )

        plotting.compare_velocity_to_measure(
            self.model,
            abs_coords,
            source=self.src.coordinates.data,
            receiver=self.rec.coordinates.data,
        )

        return mse
