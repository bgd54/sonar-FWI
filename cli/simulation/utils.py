import tqdm
import time
import numpy as np
import numpy.typing as npt

from mpi4py import MPI
from enum import Enum
from dataclasses import dataclass

from typing import Optional, Union, Tuple

Float = Union[float, np.floating]


@dataclass
class FlatBottom:
    """Flat bottom"""

    obstacle: bool = False


@dataclass
class EllipsisBottom:
    """Ellipsis shaped bottom"""

    obstacle: bool = False


@dataclass
class CircleBottom:
    """Circle shaped bottom"""

    cx: float  # m
    cy: float  # m
    r: float  # m


Bottom_type = FlatBottom | EllipsisBottom | CircleBottom


def gen_velocity_profile(
    bottom: Bottom_type,
    domain_dims: Tuple[int, int],
    spatial_dist: float = 0.1,
    v_wall: float = 3.24,
    v_water: float = 1.5,
) -> npt.NDArray[np.float32]:
    """Generate velocity profile for the simulation."""
    vp = np.full(domain_dims, v_water, dtype=np.float32)
    match bottom:
        case FlatBottom(obstacle):
            y_wall = max(
                int(domain_dims[1] * 0.8), round(domain_dims[1] - 5 / spatial_dist)
            )
            vp[:, y_wall:] = v_wall
            if obstacle:
                r = vp.shape[0] / 100
                for i in tqdm.tqdm(range(1, 101, 2)):
                    a, b = vp.shape[0] / 100 * i, y_wall
                    y, x = np.ogrid[-a : vp.shape[0] - a, -b : vp.shape[1] - b]
                    vp[x * x + y * y <= r * r] = v_wall
        case EllipsisBottom(obstacle):
            nx = domain_dims[0]
            nz = domain_dims[1]
            wall = round(nx * 0.02)
            a = round((nx - wall) / 2)
            b = round((nz - wall) / 2)
            offs = round(wall / 2)
            x = np.arange(0, vp.shape[0])
            y = np.arange(0, vp.shape[1])
            mask = (y[np.newaxis, :] - offs - b) ** 2 / b**2 + (
                x[:, np.newaxis] - offs - a
            ) ** 2 / a**2 > 1
            vp[mask] = v_wall
            if obstacle:
                r = vp.shape[0] / 100
                ox = np.arange(offs, 2 * a + offs + 1, 2 * a / 50)
                oy = np.sqrt(1 - (ox - a - offs) ** 2 / a**2) * b + offs + b
                for oxx, oyy in tqdm.tqdm(zip(ox, oy)):
                    mask = (y[np.newaxis, :] - oyy) ** 2 + (
                        x[:, np.newaxis] - oxx
                    ) ** 2 < r**2
                    vp[mask] = v_wall
            vp[offs:-offs, :b] = v_water
        case CircleBottom(cx, cy, r):
            ox = cx / spatial_dist
            oy = cy / spatial_dist
            r = round(r / spatial_dist)
            x = np.arange(0, vp.shape[0])
            y = np.arange(0, vp.shape[1])
            mask = (y[np.newaxis, :] - oy) ** 2 + (x[:, np.newaxis] - ox) ** 2 > r**2
            vp[mask] = v_wall

    return vp


def positions_line(
    stop_x: Float,
    posy: Float,
    n: int = 128,
):
    """
    Generate points in a line

    Args:
        stop_x (float): x position of last point domain.
        nr (int): Number of points to generate.
        posy (float): Absolute depth of the points

    Returns:
        Absolute positions of receivers
    """
    start = [0, posy]
    end = [stop_x, posy]
    return np.linspace(start, end, num=n)


def positions_half_circle(
    r: Float,
    c_x: Float,
    c_y: Float,
    n: int = 181,
):
    """
    Generate points in a half circle

    Args:
        r (float): Radius of the circle
        c_x (float): x position of the center of the circle
        c_y (float): y position of the center of the circle
        n (int): Number of points to generate.
    """
    res = np.zeros((n, 2))
    res[:, 0] = np.cos(np.deg2rad(np.linspace(0, 180, n))) * r + c_x
    res[:, 1] = np.sin(np.deg2rad(np.linspace(0, 180, n))) * r + c_y
    return res


def num_iter_for_distance(
    distance: float,
    dt: float,
    v_env: float,
):
    """
    Return the number of iterations needed to travel a certain distance

    Args:
        distance (float): Distance to travel in [m]
        v_env (float): Environment velocity in [km/s] if dt in [ms] or [m/s] if dt in [s]
        v_env (float): Time step in [ms] or [s]
    """
    return distance * 2 / v_env / dt


def object_distance(
    step: int,
    dt: float,
    v_env: float,
):
    """
    Return the distance traveled by an object in a certain number of iterations

    Args:
        step (int): Number of iterations
        dt (float): Time step in [ms] or [s]
        v_env (float): Environment velocity in [km/s] if dt in [ms] or [m/s] if dt in [s]
    """
    return ((step * dt) / 2) * v_env


def detect_echo_after(
    recording,
    timestep: float,
    ideal_signal: npt.NDArray,
    cut_ts: Optional[int] = None,
    cut_ms: Optional[float] = None,
):
    """
    Detect the echo after a certain time or timestep.
    """
    if cut_ts is None:
        assert cut_ms is not None
        cut_ts = round(cut_ms / timestep)
    correlation = np.correlate(recording[cut_ts:], ideal_signal)
    return cut_ts + correlation.argmax() - ideal_signal.shape[0]


def echo_distance(
    receiver,
    timestep: float,
    ideal_signal: npt.NDArray,
    max_latency: float,
    v_env: float,
    cut_ms: Optional[float] = None,
    cut_ts: Optional[int] = None,
) -> tuple[float, float]:
    """
    Calculate the distance of the object from the receiver.

    Args:
        receiver (npt.NDArray): Received signal.
        timestep (float): Timestep of the simulation.
        signal (npt.NDArray): Original signal.
        v_env (float): Velocity of the water.
        cut_ms (float): Time to mask echos from.

    Returns:
        tuple[float, float]: Distance of the object from the receiver and the time of the peak.
    """
    if cut_ts is None:
        assert cut_ms is not None
        cut_ts = round(cut_ms / timestep)
    echo = detect_echo_after(receiver, timestep, ideal_signal, cut_ms=cut_ms)
    distance = object_distance(echo, timestep, v_env)
    return distance, receiver[echo]


def run_beam(
    src,
    rec,
    op,
    u,
    source_distance: float,
    time_range,
    dt: float,
    alpha: float,
    v_env: float,
):
    """
    Run a beam simulation, used for testing a single SineSource. #TODO: single??

    Args:
        model (Model): Model to run the simulation on.
        src (SineSource): Source to use.
        rec (Receiver): Receiver to use.
        op (Operator): Operator to use.
        u (TimeFunction): TimeFunction to use.
        source_distance (float): Distance between sources.
        time_range (TimeAxis): TimeAxis to use.
        dt (float): Timestep of the simulation.
        alpha (float): Angle of the beam.
        v_env (float): Velocity of the sound in the medium.

    Returns:
        tuple[npt.NDArray, float]: Recorded signal and the maximum latency.
    """
    start_time = time.time()
    ns = src.coordinates.data.shape[0]
    if alpha <= 90:
        max_latency = (
            np.cos(np.deg2rad(alpha)) * ((ns - 1) * source_distance / v_env) / dt
        )
    elif alpha > 90:
        max_latency = np.cos(np.deg2rad(alpha)) * (source_distance / v_env) / dt
    for i in range(ns):
        latency = -np.cos(np.deg2rad(alpha)) * (i * source_distance / v_env)
        src.data[:, i] = np.roll(
            np.array(src.data[:, i]), int(latency / dt + max_latency)
        )
    u.data.fill(0)
    op(time=time_range.num - 2, dt=dt)
    print(f"Simulation took {time.time() - start_time} seconds")
    return rec.data


def calculate_coordinates(
    rec_pos,
    angle=[65],
    distance=[26],
    amplitude=[2.3169e-09],
):
    """
    Calculate the coordinates of the object.

    Args:
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


def calculate_coordinates_from_pos(
    rec_pos: tuple[float, float],
    angle: npt.NDArray,
    distance: npt.NDArray,
) -> npt.NDArray:
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
