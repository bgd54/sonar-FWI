import math
import os
import time
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
from examples.seismic import (Model, Receiver, RickerSource, TimeAxis,
                              WaveletSource, plot_shotrecord, plot_velocity)
from scipy.signal import find_peaks, peak_prominences


def srcPositions(cx: float, cy: float, alpha: float, ns: int,
                 sdist: float) -> np.typing.NDArray:
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


def srcPositionsInDomain(
        domain_size: tuple[np.float64, np.float64],
        posx: float = 0.5,
        posy: float = 0.0,
        source_distance: float = .1,
        ns: int = 128,
        angle: float = 90) -> tuple[np.typing.NDArray, tuple[float, float]]:
    # angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
    #posx and posy are relative positions 0 - left, 1 - right (or top-bottom)
    cx = domain_size[0] * posx
    if posy == 0.0:
        cy = (ns - 1) / 2 * source_distance
    else:
        cy = domain_size[1] * posy
    return srcPositions(cx, cy, angle, ns, source_distance), (cx, cy)


class SineSource(WaveletSource):

    @property
    def wavelet(self):
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = (np.pi * self.f0 * (self.time_values - t0))
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
                 v_water=1.4967):
    # Set time range, source, source coordinates and receiver coordinates
    t0 = 0.  # Simulation starts a t=0
    #tn - Simulation lasts tn milliseconds
    dt = model.critical_dt  # Time step from model grid spacing
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    #f0 Source peak frequency is 5 MHz (5000 kHz)
    #ns number of sources
    nr = ns
    # number of receivers
    depth = 1
    wavelength = v_water / f0  # 0.03 cm
    source_distance = wavelength / 8
    source_angle = angle  # angle of the sources to the water surface (0° - 180°) 90° means sources are parallel with the water surface
    #posx and posy are relative positions 0 - left, 1 - right (or top-bottom)
    pos, center_pos = srcPositionsInDomain(model.domain_size,
                                           posx=posx,
                                           posy=posy,
                                           angle=source_angle,
                                           ns=ns,
                                           source_distance=source_distance)
    src = SineSource(name='src',
                     grid=model.grid,
                     f0=f0,
                     time_range=time_range,
                     npoint=ns)

    src.coordinates.data[:, -1] = depth
    src.coordinates.data[:] = pos[:]

    rec = Receiver(name='rec',
                   grid=model.grid,
                   npoint=nr,
                   time_range=time_range)

    rec.coordinates.data[:, -1] = depth
    rec.coordinates.data[:] = pos[:]
    return src, rec, time_range, center_pos


def objDistance(receiver, timestep: float,
                v_env: float) -> Tuple[float, float]:
    #v_env in m/s
    #timestep in s
    # distance in m
    cut = 600
    x = receiver[cut:]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    first_peak = cut + peaks[
        (prominences - np.average(prominences)) > np.std(prominences)][0]
    distance = (((first_peak * timestep) / 2) * v_env)
    return distance, x[first_peak - cut]


def run_positions_angles(model, v_water, posx=[0.5], posy=0.0, angle=[90]):
    if np.size(posy) > 1 and np.size(posx) != np.size(posy):
        print(
            "error, posx and posy arrays must be same length, or posy length 1"
        )
        return
    if np.size(posy) == 1:
        posyarr = np.ones(np.shape(posx)) * posy
    distances = np.zeros((np.size(posx), np.size(angle)))
    amplitudes = np.zeros((np.size(posx), np.size(angle)))
    print(np.shape(distances))
    res = np.zeros((len(angle), 4606, 128))
    for i in range(np.size(posx)):
        for j in range(np.size(angle)):
            start = time.time()
            src, rec, time_range = setup_domain(model,
                                                tn=5,
                                                ns=128,
                                                f0=50,
                                                posx=posx[i],
                                                posy=posyarr[i],
                                                angle=angle[j])
            u = TimeFunction(name="u",
                             grid=model.grid,
                             time_order=2,
                             space_order=2)
            # Set symbolics of the operator, source and receivers:
            pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
            stencil = Eq(u.forward, solve(pde, u.forward))
            src_term = src.inject(field=u.forward,
                                  expr=src * model.critical_dt**2 / model.m)
            rec_term = rec.interpolate(expr=u)

            op = Operator([stencil] + src_term + rec_term,
                          subs=model.spacing_map)
            # Run the operator for `(nt-2)` time steps:
            op(time=time_range.num - 2, dt=model.critical_dt)
            res[j] = rec.data
            result = objDistance(np.average(rec.data, axis=1),
                                 model.critical_dt, v_water)
            distances[i, j] = result[0]
            amplitudes[i, j] = result[1]
            print(f"Iteration took: {time.time() - start}")
    return distances, amplitudes, res
    #  return distances, amplitudes


def run_positions_angles2(model,
                          src,
                          rec,
                          op,
                          u,
                          v_env,
                          source_distance,
                          time_range,
                          posx=[0.5],
                          posy=[0.0],
                          angle=[90]):
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

            pos, _ = srcPositionsInDomain(model.domain_size,
                                       posx=px,
                                       posy=py,
                                       angle=alpha,
                                       ns=src.coordinates.data.shape[0],
                                       source_distance=source_distance)
            src.coordinates.data[:] = pos[:]
            rec.coordinates.data[:] = pos[:]
            u.data.fill(0)

            # Run the operator for `(nt-2)` time steps:
            op(time=time_range.num - 2, dt=model.critical_dt)
            res[j] = rec.data
            result = objDistance(np.average(rec.data, axis=1),
                                 model.critical_dt, v_env)
            distances[i, j] = result[0]
            amplitudes[i, j] = result[1]
            print(f"Iteration took: {time.time() - start}")
    return distances, amplitudes, res


def calculate_coordinates(domain_size,
                          rec_pos: npt.NDArray,
                          angle=[65],
                          distance=[26],
                          amplitude=[2.3169e-09]):
    if np.size(amplitude) != np.size(distance):
        print("error, angle and distance arrays must be same length")
        return
    results = np.zeros((len(rec_pos), np.size(angle), 2))
    for i, pos in enumerate(rec_pos):
        for j, alpha in enumerate(angle):
            sx = pos[0]
            sy = pos[1]
            results[i, j,
                    0] = sx - np.cos(alpha * np.pi / 180) * distance[i, j]
            results[i, j,
                    1] = sy + np.sin(alpha * np.pi / 180) * distance[i, j]
    return results
