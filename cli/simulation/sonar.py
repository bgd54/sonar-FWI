"""This module provides the sonar class for the water tank project."""
import math
import tqdm
import copy

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
from examples.seismic import Model, TimeAxis, Receiver, WaveletSource

from simulation import plotting, utils
from simulation.sources import SineSource, MultiFrequencySource, GaborSource


class Sonar:
    """
    Sonar class with optional source type, default is a sine source.

    Args:
        domain_size (Tuple[float, float]): Size of the domain in meters.
        f_critical (float): Critical frequency of the signal.
        v_water (float): Speed of sound in the water.
        velocity_profile (Union[npt.NDArray, utils.Bottom_type]): Velocity profile of the domain.
        tn (float): Total time of the simulation.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float],
        f_critical: float,
        v_water: float,
        velocity_profile: Union[npt.NDArray, utils.Bottom_type],
        ns: int = 128,
        source_distance: float = 0.002,
        space_order: Optional[int] = None,
        time_order: Optional[int] = None,
        tn: Optional[float] = None,
        dt: Optional[float] = None,
        spatial_dist: Optional[float] = None,
        nbl: Optional[float] = None,
    ) -> None:
        self.f0 = f_critical
        self.space_order = space_order if space_order is not None else 2
        self.time_order = time_order if time_order is not None else 2
        self.spatial_dist = (
            spatial_dist
            if spatial_dist is not None
            else round(v_water / self.f0 / 3, 3) / 2
        )
        self.nbl = nbl if nbl is not None else (ns - 1) / 2 * source_distance / dt
        self.dt = dt if dt is not None else self.spatial_dist / 10
        self.domain_size = domain_size
        self.v_env = v_water
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
        if dt is None:
            self.model = Model(
                vp=vp,
                origin=(0.0, 0.0),
                spacing=(self.spatial_dist, self.spatial_dist),
                shape=domain_dims,
                space_order=space_order,
                nbl=self.nbl,
                bcs=self.boundary,
            )
        else:
            self.model = Model(
                vp=vp,
                origin=(0.0, 0.0),
                spacing=(self.spatial_dist, self.spatial_dist),
                shape=domain_dims,
                space_order=space_order,
                nbl=self.nbl,
                bcs=self.boundary,
                dt=dt,
            )
        if tn is None:
            max_distance = math.sqrt((domain_size[0] / 2) ** 2 + domain_size[1] ** 2)
            tn = max_distance * 2 / v_water + 5
        self.time_range = TimeAxis(start=0.0, stop=tn, step=self.model.critical_dt)
        self.u = None
        self.usave = None
        self.src = None
        self.rec = None
        self.op = None

    def set_source(
        self,
        src: WaveletSource = None,
        rec: Receiver = None,
    ) -> None:
        """
        Set the source and receiver for the simulation. If no source and receiver is given, use SineSource as default.

        Args:
            src (WaveletSource): Source object.
            rec (Receiver): Receiver object.
            source_distance (float): Distance between sources.
            ns (int): Number of sources.
        """
        if src is None and rec is None:
            assert source_distance is not None and ns is not None
            cy = (ns - 1) / 2 * source_distance
            src_coord = np.array(
                [(self.domain_size[0] - source_distance * ns) / 2, cy]
            ) + utils.positions_line(
                stop_x=ns * source_distance, posy=source_distance, n=ns
            )
            self.src = GaborSource(
                name="src",
                grid=self.model.grid,
                npoint=ns,
                f0=self.f0,
                time_range=self.time_range,
                coordinates_data=src_coord,
            )
            self.rec = Receiver(
                name="rec",
                grid=self.model.grid,
                time_range=self.time_range,
                npoint=ns,
                coordinates=src_coord,
            )
        else:
            assert src is not None and rec is not None
            self.src = copy.deepcopy(src)
            self.rec = copy.deepcopy(rec)

    def finalize(
        self,
        snapshot_delay: Optional[float] = None,
    ) -> None:
        """
        Finalize the simulation by creating the operator and setting the source and receiver.

        Args:
            snapshot_delay (float): Delay between snapshots, default is None, which means no snapshots are saved.
        """
        assert self.src is not None and self.rec is not None
        self.u = TimeFunction(
            name="u",
            grid=self.model.grid,
            time_order=self.time_order,
            space_order=self.space_order,
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
                time_order=self.time_order,
                space_order=self.space_order,
                save=nsnaps,
                time_dim=time_subsampled,
            )
            save_stencil.append(Eq(self.usave, self.u))

        self.op = Operator(
            [stencil] + save_stencil + src_term + rec_term, subs=self.model.spacing_map
        )
