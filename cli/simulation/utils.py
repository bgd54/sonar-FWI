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


def correlate(
    recording: npt.NDArray, ideal_signal: npt.NDArray, start_iter: int = 5000
) -> int:
    """
    Correlate the recording with the ideal signal and return the index of the peak.

    Args:
        recording: The recording of a single receiver to correlate with the ideal signal.
        ideal_signal: The ideal signal.
        start_iter: The index to start the search from.

    Returns:
        The index of the peak.
    """
    assert recording.shape[0] > 5000
    correlate = np.correlate(recording[start_iter:], ideal_signal, mode="same")
    peak = start_iter + correlate.argmax()
    return peak


def iters2dist(
    stop_iter: int, dt: float, v_env: float = 1500, start_iter: int = 0
) -> float:
    """
    Convert the index to distance.

    Args:
        end_iter: The index of a time iteration.
        dt: The time step (s).
        v_env: The velocity of the environment (m/s).
        start_iter: The index when the signal is sent.

    Returns:
        The distance (m).
    """
    return (stop_iter - start_iter) * dt * v_env


def rec2coords(
    recording: npt.NDArray,
    receiver_coords: npt.NDArray,
    ideal_signal: npt.NDArray,
    angle: int,
    dt: float,
    start_iter: int = 5000,
) -> npt.NDArray:
    """
    Convert the recording to coordinates.

    Args:
        recording: The recording of a single receiver.
        receiver_coords: The coordinates of the receiver.
        angle: The angle of the beam.
        dt: The time step (s).
    """
    assert recording.shape[0] > 5000 and recording.shape[1] == receiver_coords.shape[0]
    ns = recording.shape[1]
    coordinates = np.zeros((ns, 2))
    for i in range(ns):
        start_time = np.argmax(recording[:start_iter, i])
        peak = correlate(recording[:, i], ideal_signal, start_iter)
        distance = iters2dist(peak - start_time, dt, start_iter=start_time) / 2
        rec_coords = receiver_coords[i]
        coordinates[i, 0] = rec_coords[0] + distance * np.cos(np.deg2rad(angle))
        coordinates[i, 1] = rec_coords[1] + distance * np.sin(np.deg2rad(angle))
    return np.mean(coordinates, axis=0)
