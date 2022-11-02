import math
import time
from enum import Enum

import numpy as np
import numpy.typing as npt
from examples.seismic import Receiver, TimeAxis, WaveletSource
from scipy.signal import find_peaks, peak_prominences


class Bottom(str, Enum):
    flat = "flat"
    ellipsis = "ellipsis"
    circle = "circle"


def find_exp(number: float) -> int:
    """Find the exponent of a number.

    Args:
        number (float): Number to find the exponent of.

    Returns:
        int: Exponent of the number.
    """
    return math.floor(math.log10(number))


def src_positions(cx: float, cy: float, alpha: float, ns: int,
                  sdist: float) -> np.typing.NDArray:
    """
    Create source positions.

    Args:
        cx (float): x coordinate of the center of the source array.
        cy (float): y coordinate of the center of the source array.
        alpha (float): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
        ns (int): Number of sources.
        sdist (float): Distance between sources.

    Returns:
        np.typing.NDArray: Source positions.
    """
    assert alpha >= 0 and alpha < 180
    assert ns > 0
    dx = sdist * math.sin(math.pi / 180 * alpha)
    dy = sdist * math.cos(math.pi / 180 * alpha)

    res = np.zeros((ns, 2))
    res[:, 0] = np.linspace(cx - dx * (ns - 1) / 2,
                            cx + dx * (ns - 1) / 2,
                            num=ns)
    res[:, 1] = np.linspace(cy - dy * (ns - 1) / 2,
                            cy + dy * (ns - 1) / 2,
                            num=ns)
    return res


def src_positions_in_domain(
    domain_size: tuple[np.float64, np.float64],
    posx: float = 0.5,
    posy: float = 0.0,
    source_distance: float = 0.1,
    ns: int = 128,
    angle: float = 90,
) -> tuple[np.typing.NDArray, tuple[float, float]]:
    """
    Create the source positions in the domain.

    Args:
        domain_size (tuple[np.float64, np.float64]): Size of the domain in x and y direction.
        posx (float): Relative position of the center of the source array in x direction.
        posy (float): Relative position of the center of the source array in y direction.
        source_distance (float): Distance between sources.
        ns (int): Number of sources.
        angle (float): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface

    Returns:
        tuple[np.typing.NDArray, tuple[float, float]]: Source positions and the center of the source array.
    """
    cx = domain_size[0] * posx
    if posy == 0.0:
        cy = (ns - 1) / 2 * source_distance
    else:
        cy = domain_size[1] * posy
    return src_positions(cx, cy, angle, ns, source_distance), (cx, cy)


class SineSource(WaveletSource):
    """
    A source object that encapsulates everything necessary for injecting a
    sine source into the computational domain.

    Returns:
        The source term that will be injected into the computational domain.
    """

    @property
    def wavelet(self):
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = np.pi * self.f0 * (self.time_values - t0)
        wave = a * np.sin(r) + a * np.sin(3 * (r + np.pi) / 4)
        wave[np.searchsorted(self.time_values, 4 * 2 / self.f0):] = 0
        return wave


def setup_domain(model,
                 tn=0.05,
                 ns=128,
                 f0=5000,
                 posx=0.5,
                 posy=0.0,
                 angle=90,
                 v_env=1.4967):
    """
    Setup the domain.

    Args:
        model (Model): Model of the domain.
        tn (float): End time of the simulation.
        ns (int): Number of sources.
        f0 (float): Center frequency of the signal.
        posx (float): Relative position of the center of the source array in x direction.
        posy (float): Relative position of the center of the source array in y direction.
        angle (float): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
        v_env (float): Velocity of the water.

    Returns:
        Source and receiver, center of the source array and wavelength.
    """
    t0 = 0.0
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    nr = ns
    wavelength = v_env / f0
    sdist = wavelength / 8
    sangle = angle
    src_positions, src_center = src_positions_in_domain(
        model.domain_size,
        posx=posx,
        posy=posy,
        source_distance=sdist,
        ns=ns,
        angle=sangle,
    )

    src = SineSource(name="src",
                     grid=model.grid,
                     f0=f0,
                     time_range=time_range,
                     npoint=ns)
    src.coordinates.data[:] = src_positions
    rec = Receiver(
        name="rec",
        grid=model.grid,
        time_range=time_range,
        npoint=nr,
        coordinates=src.coordinates.data,
    )

    return src, rec, time_range, src_center, sdist


def num_iter_for_distance(distance: float, dt: float, v_env: float):
    # distance in [m]
    # v_env in [km/s] if dt in [ms]
    # or
    # v_env in [m/s] if dt in [s]
    return distance * 2 / v_env / dt


def object_distance_iter(step: int, dt: float, v_env: float):
    # step: dimensionless
    # v_env in [km/s] if dt in [ms]
    # or
    # v_env in [m/s] if dt in [s]
    # out distance in [m]
    return ((step * dt) / 2) * v_env


def find_first_peak(recording, timestep: float, v_env: float) -> int:
    peaks, _ = find_peaks(recording)
    prominences = peak_prominences(recording, peaks)[0]
    return peaks[(prominences -
                  np.average(prominences)) > np.std(prominences)][0]


def first_peak_after(recording,
                     timestep: float,
                     v_env: float,
                     cut_iter: int = None,
                     cut_ms: float = None):
    if cut_iter is None:
        cut_iter = round(cut_ms / timestep)
    return cut_iter + find_first_peak(recording[cut_iter:], timestep, v_env)


def object_distance(receiver,
                    timestep: float,
                    v_env: float,
                    cut_ms: float = 2.0) -> tuple[float, float]:
    """
    Calculate the distance of the object from the receiver.

    Args:
        receiver (Receiver): Receiver of the signal.
        timestep (float): Timestep of the simulation.
        v_env (float): Velocity of the water.

    Returns:
        tuple[float, float]: Distance of the object from the receiver and the time of the peak.
    """
    first_peak = first_peak_after(receiver, timestep, v_env, cut_ms=cut_ms)
    distance = object_distance_iter(first_peak, timestep, v_env)
    print(f"distance {distance} m <- {v_env} / 2 * {first_peak} * {timestep}")
    return distance, receiver[first_peak]


def setup_beam(src, rec, u, source_distance, center_pos, alpha):
    pos = src_positions(center_pos[0], center_pos[1], alpha,
                        src.coordinates.data.shape[0], source_distance)
    src.coordinates.data[:] = pos[:]
    rec.coordinates.data[:] = pos[:]
    u.data.fill(0)


def run_beam(model, src, rec, op, u, source_distance, time_range, center_pos,
             alpha):
    setup_beam(src, rec, u, source_distance, center_pos, alpha)

    # Run the operator for `(nt-2)` time steps:
    print(f"time = {time_range.num} dt = {model.critical_dt}")
    op(time=time_range.num - 2, dt=model.critical_dt)


def run_positions_angles(
    model,
    src,
    rec,
    op,
    u,
    v_env,
    source_distance,
    time_range,
    centers,
    angle=[90],
):
    """
    Run the simulation for different source positions and angles.

    Args:
        model (Model): Model of the domain.
        src (SineSource): Source of the signal.
        rec (Receiver): Receiver of the signal.
        op (Operator): Operator of the simulation.
        u (TimeFunction): TimeFunction of the simulation.
        v_env (float): Velocity of the water.
        source_distance (float): Distance between sources.
        time_range (TimeAxis): TimeAxis of the simulation.
        centers (list[tuple[float, float]]): Absolute position of the center of the source array.
        angle (list[float]): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface

    Returns:
        distances (list[float]): Distances of the object from the receiver.
        amplitudes (list[float]): Amplitudes of the signal at the receiver.
        res
    """
    distances = np.zeros((len(centers), np.size(angle)))
    amplitudes = np.zeros((len(centers), np.size(angle)))
    print(np.shape(distances))
    res = np.zeros((len(angle), rec.data.shape[0], rec.data.shape[1]))
    for i, center_pos in enumerate(centers):
        for j, alpha in enumerate(angle):
            start = time.time()

            run_beam(model, src, rec, op, u, source_distance, time_range,
                     center_pos, alpha)

            res[j] = rec.data
            result = object_distance(np.average(rec.data, axis=1),
                                     model.critical_dt, v_env)
            distances[i, j] = result[0]
            amplitudes[i, j] = result[1]
            print(f"Iteration took: {time.time() - start}")
    return distances, amplitudes, res


def run_angles(
    model,
    src,
    rec,
    op,
    u,
    source_distance,
    time_range,
    center,
    angle=[90],
):
    """
    Run the simulation for different angles.

    Args:
        model (Model): Model of the domain.
        src (SineSource): Source of the signal.
        rec (Receiver): Receiver of the signal.
        op (Operator): Operator of the simulation.
        u (TimeFunction): TimeFunction of the simulation.
        source_distance (float): Distance between sources.
        time_range (TimeAxis): TimeAxis of the simulation.
        center (tuple[float,float]): Absolute position of the center of the source array.
        angle (list[float]): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface

    Returns:
        res: Receiver data fom beam simulations
    """
    res = np.zeros((angle.shape[0], rec.data.shape[0], rec.data.shape[1]))
    for j, alpha in enumerate(angle):
        start = time.time()
        run_beam(model, src, rec, op, u, source_distance, time_range, center,
                 alpha)
        res[j] = rec.data
        print(f"Iteration took: {time.time() - start}")
    return res


def calculate_coordinates(domain_size,
                          rec_pos,
                          angle=[65],
                          distance=[26],
                          amplitude=[2.3169e-09]):
    """
    Calculate the coordinates of the object.

    Args:
        domain_size (tuple[float]): Size of the domain.
        rec_pos (tuple[float]): Position of the receiver.
        angle (list[float]): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
        distance (list[float]): Distance of the object from the receiver.
        amplitude (list[float]): Amplitude of the signal at the receiver.

    Returns:
        coordinates (list[float]): Coordinates of the object.
    """
    if np.size(amplitude) != np.size(distance):
        print("error, angle and distance arrays must be same length")
        return
    coordinates = np.zeros((len(rec_pos), np.size(angle), 2))
    for i, pos in enumerate(rec_pos):
        for j, alpha in enumerate(angle):
            sx = pos[0]
            sy = pos[1]
            coordinates[i, j,
                        0] = sx - np.cos(alpha * np.pi / 180) * distance[i, j]
            coordinates[i, j,
                        1] = sy + np.sin(alpha * np.pi / 180) * distance[i, j]
    return coordinates


def calculate_coordinates_from_pos(rec_pos: tuple[float,
                                                  float], angle: npt.NDArray,
                                   distance: npt.NDArray) -> npt.NDArray:
    """
    Calculate the absolute coordinates of the object.

    Args:
        rec_pos (tuple[float]): Position of the receiver.
        angle (list[float]): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
        distance (list[float]): Distance of the object from the receiver.

    Returns:
        coordinates (list[float]): Coordinates of the object.
    """
    coordinates = np.zeros((np.size(angle), 2))
    for j, alpha in enumerate(angle):
        sx = rec_pos[0]
        sy = rec_pos[1]
        coordinates[j, 0] = sx - np.cos(alpha * np.pi / 180) * distance[j]
        coordinates[j, 1] = sy + np.sin(alpha * np.pi / 180) * distance[j]
    return coordinates
