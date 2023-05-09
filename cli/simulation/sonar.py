"""This module provides the sonar class for the water tank project."""
import math
import tqdm

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
from examples.seismic import Model, TimeAxis, Receiver, WaveletSource

from simulation import plotting, utils


class Sonar:
    """Sonar class for the water tank project."""

    def __init__(
        self,
        size_x: int,
        size_y: int,
        f0: float,
        v_env: float,
        ns: int,    # src_pos, rec_pos
        posx: float,# src_pos, rec_pos
        posy: float,# src_pos, rec_pos
        bottom: utils.Bottom,
        source_distance: float,# src_pos, rec_pos
        snapshot_delay: float = 0.0,
        obstacle: bool = False,
        r: float = 28.0,
        rec_positions: Optional[npt.NDArray] = None,
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
        shape = (self.size_x, self.size_y)
        spacing = (self.spatial_dist, self.spatial_dist)
        origin = (0.0, 0.0)
        posy = posy if posy != 0.0 else ((ns - 1) / 2 * self.sdist) / size_y
        self.model = Model(
            vp=self.set_bottom(bottom, cx=posx, cy=posy, r=r),
            origin=origin,
            spacing=spacing,
            shape=shape,
            space_order=2,
            nbl=10,
            bcs="damp",
        )
        travel_distance = math.sqrt(
            (size_x * (posx if posx >= 0.5 else 1 - posx)) ** 2
            + (size_y * (posy if posy >= 0.5 else 1 - posy)) ** 2
        )
        self.tn = travel_distance * 2 / v_env + 5
        self.time_range = TimeAxis(start=0.0, stop=self.tn, step=self.model.critical_dt)
        (self.src, self.rec, self.center_pos) = utils.setup_sources_and_receivers(
            self.model,
            time_range=self.time_range,
            ns=ns,
            f0=self.f0,
            posx=posx,
            posy=posy,
            sdist=self.sdist,
        )
        if rec_positions is not None:
            self.rec = Receiver(
                name="rec",
                grid=self.model.grid,
                time_range=self.time_range,
                npoint=rec_positions.shape[0],
                coordinates=rec_positions,
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
        self.snapshot_delay = snapshot_delay
        self.setup_equations()


    def setup_equations(self):
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
        if self.snapshot_delay > 0.0:
            snapshot_delay_iter = math.ceil(self.snapshot_delay / self.model.critical_dt)
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
        r: Optional[float] = None,
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
            assert r is not None
            ox = int(cx * self.size_x)
            oy = int(cy * self.size_y)
            r = round(r / self.spatial_dist)
            x = np.arange(0, v.shape[0])
            y = np.arange(0, v.shape[1])
            mask = (y[np.newaxis, :] - oy) ** 2 + (x[:, np.newaxis] - ox) ** 2 > r**2
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
            rec_pos=[self.center_pos],
            angle=angles,
            distance=results[0],
            amplitude=results[1],
        )
        assert res2 is not None

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
            v_env=self.v_env,
        )

    def parse_and_plot(self, angles, recordings):
        distances = np.zeros(angles.shape)
        for i, (_, rec) in tqdm.tqdm(enumerate(zip(angles, recordings))):
            distances[i], _ = utils.echo_distance(
                np.average(rec, axis=1),
                self.model.critical_dt,
                self.src.signal_packet,
                self.v_env,
            )

        abs_coords = utils.calculate_coordinates_from_pos(
            rec_pos=self.center_pos, angle=angles, distance=distances
        )

        plotting.compare_velocity_to_measure(
            self.model,
            abs_coords,
            source=self.src.coordinates.data,
            receiver=self.rec.coordinates.data,
        )

class Sonar_v2:
    """Sonar simulation class for water tank simulations with optinal source type."""

    def __init__(
        self,
        domain_size: Tuple[float, float],
        f_critical: float,
        v_water: float,
        velocity_profile: Union[npt.NDArray, utils.Bottom_type],
        tn: float,
        spatial_dist: float = 0.0,
        dt: float = 0.0,
    ) -> None:
        self.f0 = f_critical
        if spatial_dist == 0.0:
            self.spatial_dist = round(v_water / self.f0 / 3, 3)
        else:
            self.spatial_dist = spatial_dist
        domain_dims = (
            round(domain_size[0] / self.spatial_dist),
            round(domain_size[1] / self.spatial_dist),
        )
        vp = (
            velocity_profile
            if isinstance(velocity_profile, np.ndarray)
            else utils.gen_velocity_profile(
                velocity_profile, domain_dims, self.spatial_dist, v_water=v_water
            )
        )
        self.model = Model(
            vp=vp,
            origin=(0.0, 0.0),
            spacing=(self.spatial_dist, self.spatial_dist),
            shape=domain_dims,
            space_order=2,
            nbl=10,
            bcs="damp",
            dt = dt,
        )
        self.time_range = TimeAxis(start=0.0, stop=tn, step=self.model.critical_dt) 
        self.u = None
        self.usave = None
        self.src = None
        self.rec = None
        self.op = None

    def set_source(self, src: WaveletSource, rec: Receiver) -> None:
        """Set the source and receiver."""
        self.src = src
        self.rec = rec

    def finalize(self, snapshot_delay: Optional[float] = None, space_order: int = 2) -> None:
        """Set up the wave equations, source and receiver terms and snapshoting"""
        assert self.src is not None and self.rec is not None
        self.u = TimeFunction(
            name="u", grid=self.model.grid, time_order=2, space_order=space_order
        )
        pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        stencil = Eq(self.u.forward, solve(pde, self.u.forward))
        src_term = self.src.inject(
            field=self.u.forward,
            expr=self.src * self.model.critical_dt**2 / self.model.m,
        )
        rec_term = self.rec.interpolate(expr=self.u)

        save_stencil = []
        if snapshot_delay is not None:
            snapshot_delay_iter = math.ceil(snapshot_delay / self.model.critical_dt)
            nsnaps = math.ceil(self.time_range.num / snapshot_delay_iter)
            time_subsampled = ConditionalDimension(
                "t_sub", parent=self.model.grid.time_dim, factor=snapshot_delay_iter
            )
            self.usave = TimeFunction(
                name="usave",
                grid=self.model.grid,
                time_order=2,
                space_order=space_order,
                save=nsnaps,
                time_dim=time_subsampled,
            )
            save_stencil.append(Eq(self.usave, self.u))

        self.op = Operator(
            [stencil] + save_stencil + src_term + rec_term, subs=self.model.spacing_map
        )


