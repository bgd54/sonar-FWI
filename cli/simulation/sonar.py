"""This module provides the sonar class for the water tank project."""
import math
import copy
import time

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Tuple, Union, Any
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
from examples.seismic import Model, TimeAxis, Receiver, WaveletSource

from simulation import plotting, utils
from simulation.sources import SineSource, MultiFrequencySource, GaborSource

SOURCE_CLASS_MAP = {
    "SineSource": SineSource,
    "GaborSource": GaborSource,
    "MultiFrequencySource": MultiFrequencySource,
}

RECEIVER_CLASS_MAP = {
    "Receiver": Receiver,
}


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

    _SONAR_MPI = False

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
        self.space_order = space_order if space_order is not None else 8
        self.time_order = time_order if time_order is not None else 2
        self.spatial_dist = (
            spatial_dist
            if spatial_dist is not None
            else round(v_water / self.f0 / 3, 6)
        )
        self.ns = ns
        self.source_distance = source_distance
        self.dt = dt if dt is not None else self.spatial_dist / 20
        self.nbl = nbl if nbl is not None else (ns - 1) / 2 * source_distance / self.dt
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
        self.model = Model(
            vp=vp,
            origin=(0.0, 0.0),
            spacing=(self.spatial_dist, self.spatial_dist),
            shape=domain_dims,
            space_order=self.space_order,
            nbl=self.nbl,
            bcs="damp",
            dt=self.dt,
            dtype=np.float64,
        )

        if self.model.grid.distributor.nprocs > 1:
            self._SONAR_MPI = True
            self._comm = self.model.grid.distributor.comm
            self._rank = self.model.grid.distributor.myrank

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
        src: Union[None, str, WaveletSource] = None,
        src_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the source for the simulation.

            Args:
                source: Either a string representing the class name, an instance of a source class, or None.
                source_args: Dictionary of arguments for the source class, used if a class name is provided.
        """
        if src is None:
            assert self.source_distance is not None and self.ns is not None
            cy = (self.ns - 1) / 2 * self.source_distance
            src_coord = np.array(
                [(self.domain_size[0] - self.source_distance * self.ns) / 2, cy]
            ) + utils.positions_line(
                stop_x=self.ns * self.source_distance,
                posy=self.source_distance,
                n=self.ns,
            )
            self.src = GaborSource(
                name="src",
                grid=self.model.grid,
                npoint=self.ns,
                f0=self.f0,
                time_range=self.time_range,
                coordinates_data=src_coord,
            )
        elif isinstance(src, str):
            src_class = SOURCE_CLASS_MAP[src]
            self.src = src_class(**src_args) if src_args else src_class()
        else:
            raise ValueError(
                "Invalid source type. Must be a string or a WaveletSource."
            )

    def set_receiver(
        self,
        rec: Union[None, str, Receiver] = None,
        rec_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the receiver for the simulation.

        Args:
            rec: Either a string representing the class name, an instance of a receiver class, or None.
            rec_args: Dictionary of arguments for the receiver class, used if a class name is provided.
        """

        if rec is None:
            assert self.source_distance is not None and self.ns is not None
            cy = (self.ns - 1) / 2 * self.source_distance
            rec_coord = np.array(
                [(self.domain_size[0] - self.source_distance * self.ns) / 2, cy]
            ) + utils.positions_line(
                stop_x=self.ns * self.source_distance,
                posy=self.source_distance,
                n=self.ns,
            )
            self.rec = Receiver(
                name="rec",
                grid=self.model.grid,
                time_range=self.time_range,
                npoint=self.ns,
                coordinates=rec_coord,
            )
        elif isinstance(rec, str):
            rec_class = RECEIVER_CLASS_MAP[rec]
            self.rec = rec_class(**rec_args) if rec_args else rec_class()
        else:
            raise ValueError("Invalid receiver type. Must be a string or a Receiver.")

    @property
    def recording(self) -> npt.NDArray[np.float64]:
        """Get the recorded signal.

        Returns:
            npt.NDArray[np.float64]: The recorded signal.
        """
        if self._SONAR_MPI:
            all_recording = self._comm.gather(self.rec.data, root=0)
            if self._rank == 0:
                all_recording = np.concatenate(all_recording, axis=1)
                return all_recording
        else:
            return copy.deepcopy(self.rec.data)

    def save_ideal_signal(self, filename: str) -> None:
        """Save the ideal signal to a file.

        Args:
            dir (None): Directory to save the file to.
        """
        if self._SONAR_MPI:
            if self._rank == 0:
                np.save(f"{filename}.npy", self.src.signal_packet)
        else:
            np.save(f"{filename}.npy", self.src.signal_packet)

    def save_recording(self, filename: str) -> None:
        """Save the recorded signal to a file.

        Args:
            dir (None): Directory to save the file to.
        """
        if self._SONAR_MPI:
            rec = self.recording
            if self._rank == 0:
                np.save(f"{filename}.npy", rec)
        else:
            np.save(f"{filename}.npy", self.recording)

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

    def run_beam(self, alpha: float) -> None:
        """Run beam simulation.

        Args:
            alpha (float): Angle of the beam.l.
        """
        for i in range(self.ns):
            self.src.data[:, i] = self.src.wavelet
        self.rec.data.fill(0)
        start_time = time.time()
        if alpha <= 90:
            max_latency = (
                np.cos(np.deg2rad(alpha))
                * ((self.ns - 1) * self.source_distance / self.v_env)
                / self.dt
            )
        elif alpha > 90:
            max_latency = (
                np.cos(np.deg2rad(alpha))
                * (self.source_distance / self.v_env)
                / self.dt
            )
        for i in range(self.ns):
            latency = -np.cos(np.deg2rad(alpha)) * (
                i * self.source_distance / self.v_env
            )
            self.src.data[:, i] = np.roll(
                np.array(self.src.data[:, i]), int(latency / self.dt + max_latency)
            )
        self.u.data.fill(0)
        self.op(time=self.time_range.num - 2, dt=self.dt)
        print(f"Simulation took {time.time() - start_time} seconds")
