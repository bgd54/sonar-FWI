import math
import time

import numpy as np
import numpy.typing as npt
from examples.seismic import Receiver, WaveletSource, TimeAxis
from scipy.signal import find_peaks, peak_prominences
from enum import Enum


class Bottom(str, Enum):
    flat = "flat"
    ellipsis = "ellipsis"


def find_exp(number: float) -> int:
    """Find the exponent of a number.

    Args:
        number (float): Number to find the exponent of.

    Returns:
        int: Exponent of the number.
    """
    return math.floor(math.log10(number))


def src_positions(
    cx: float, cy: float, alpha: float, ns: int, sdist: float
) -> np.typing.NDArray:
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
    res[:, 0] = np.linspace(cx - dx * (ns - 1) / 2, cx + dx * (ns - 1) / 2, num=ns)
    res[:, 1] = np.linspace(cy - dy * (ns - 1) / 2, cy + dy * (ns - 1) / 2, num=ns)
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
        wave[np.searchsorted(self.time_values, 4 * 2 / self.f0) :] = 0
        return wave


def setup_domain(
    model, tn=0.05, ns=128, f0=5000, posx=0.5, posy=0.0, angle=90, v_env=1.4967
):
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

    src = SineSource(name="src", grid=model.grid, f0=f0, time_range=time_range)
    src.coordinates.data[:] = src_positions
    rec = Receiver(
        name="rec",
        grid=model.grid,
        time_range=time_range,
        npoint=nr,
        coordinates=src.coordinates.data,
    )

    return src, rec, time_range, src_center, sdist


def object_distance(
    receiver, timestep: float, v_env: float, cut: int = 600
) -> tuple[float, float]:
    """
    Calculate the distance of the object from the receiver.

    Args:
        receiver (Receiver): Receiver of the signal.
        timestep (float): Timestep of the simulation.
        v_env (float): Velocity of the water.

    Returns:
        tuple[float, float]: Distance of the object from the receiver and the time of the peak.
    """
    x = receiver[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    first_peak = (
        cut + peaks[(prominences - np.average(prominences)) > np.std(prominences)][0]
    )
    #  first_peak = cut + peaks[[x[p] > 1.49e-5 for p in peaks]][0]
    distance = ((first_peak * timestep) / 2) * v_env
    return distance, x[first_peak - cut]


def run_positions_angles(
    model,
    src,
    rec,
    op,
    u,
    v_env,
    source_distance,
    time_range,
    posx=[0.5],
    posy=[0.0],
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
        posx (list[float]): Relative position of the center of the source array in x direction.
        posy (list[float]): Relative position of the center of the source array in y direction.
        angle (list[float]): Angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface

    Returns:
        distances (list[float]): Distances of the object from the receiver.
        amplitudes (list[float]): Amplitudes of the signal at the receiver.
        res
    """
    if np.size(posx) != np.size(posy):
        print("error, posx and posy arrays must be same length")
        return
    distances = np.zeros((np.size(posx), np.size(angle)))
    amplitudes = np.zeros((np.size(posx), np.size(angle)))
    print(np.shape(distances))
    res = np.zeros((len(angle), rec.data.shape[0], rec.data.shape[1]))
    for i, (px, py) in enumerate(zip(posx, posy)):
        for j, alpha in enumerate(angle):
            start = time.time()

            pos, _ = src_positions_in_domain(
                model.domain_size,
                posx=px,
                posy=py,
                angle=alpha,
                ns=src.coordinates.data.shape[0],
                source_distance=source_distance,
            )
            src.coordinates.data[:] = pos[:]
            rec.coordinates.data[:] = pos[:]
            u.data.fill(0)

            # Run the operator for `(nt-2)` time steps:
            op(time=time_range.num - 2, dt=model.critical_dt)
            res[j] = rec.data
            result = object_distance(
                np.average(rec.data, axis=1), model.critical_dt, v_env
            )
            distances[i, j] = result[0]
            amplitudes[i, j] = result[1]
            print(f"Iteration took: {time.time() - start}")
    return distances, amplitudes, res


def calculate_coordinates(
    domain_size, rec_pos, angle=[65], distance=[26], amplitude=[2.3169e-09]
):
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
            coordinates[i, j, 0] = sx - np.cos(alpha * np.pi / 180) * distance[i, j]
            coordinates[i, j, 1] = sy + np.sin(alpha * np.pi / 180) * distance[i, j]
    return coordinates
