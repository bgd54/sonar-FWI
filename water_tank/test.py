#!/bin/python3

import itertools

import fire
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from devito import ConditionalDimension, Eq, Operator, TimeFunction, solve
from examples.seismic import (Model, Receiver, RickerSource, TimeAxis,
                              plot_velocity)

from utils import srcPositions, setup_domain, run_positions_angles2, run_positions_angles, calculate_coordinates

from plotting import compare_velocity_to_measure

from scipy.signal import find_peaks, peak_prominences


def init_circle(v: npt.NDArray, ox: int, oy: int, r: int, v_inner: float,
                v_wall: float):
    # r in index
    # ox, oy in index
    v.fill(v_inner)
    for i, j in itertools.product(range(v.shape[0]), range(v.shape[1])):
        rx = ox - i
        ry = oy - j
        if (rx**2 + ry**2) > r**2:
            v[i, j] = v_wall


def source_depth_m(num_sources: int, source_distance_m: float) -> float:
    return source_distance_m * (num_sources - 1) / 2


def source_depth_idx(num_sources: int, source_distance_m: float,
                     spacing_y: float) -> int:
    return (int)(source_depth_m(num_sources, source_distance_m) / spacing_y)


def main():
    nx = 601
    nz = 301
    nb = 10
    shape = (nx, nz)
    spacing = (.01, .01)  # in m
    origin = (0., 0.)

    #constants used in simulation
    v_water = 1.4967  # v in distilled water 1496.7 m/s
    v_glass = 5.64  # v in glass 5640 m/s
    v_obj = 3.24  # v in gold 3240 m/s
    f0 = 50  # Source peak frequency in kHz
    source_distance_m = v_water / f0 / 8  #???
    num_sources = 128

    # Define a velocity profile. The velocity is in km/s
    v = np.full(shape, v_water, dtype=np.float32)

    init_circle(
        v, 300, source_depth_idx(num_sources, source_distance_m, spacing[1]),
        nz - 10 - source_depth_idx(num_sources, source_distance_m, spacing[1]),
        v_water, v_glass)

    model = Model(vp=v,
                  origin=origin,
                  shape=shape,
                  spacing=spacing,
                  space_order=2,
                  nbl=nb,
                  bcs="damp")
    src, rec, time_range, center_pos = setup_domain(
        model,
        tn=5,
        ns=num_sources,
        f0=f0,
        posx=.5,
        posy=0.0,  # hack for source_depth_m / domain_size
        v_water=v_water)
    print(model.critical_dt)
    plot_velocity(model,
                  source=src.coordinates.data,
                  receiver=rec.coordinates.data)
    # create the operator
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
    # Set symbolics of the operator, source and receivers:
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))
    src_term = src.inject(field=u.forward,
                          expr=src * model.critical_dt**2 / model.m)
    rec_term = rec.interpolate(expr=u)

    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

    # run on angles

    angles = np.arange(5, 31)
    results = run_positions_angles2(model,
                                    src,
                                    rec,
                                    op,
                                    u,
                                    v_water,
                                    source_distance_m,
                                    time_range,
                                    posx=[0.5],
                                    posy=[0.0],
                                    angle=angles)
    print(results)

    #  results2 = run_positions_angles(model, v_water, posx=[0.5], angle=np.arange(5, 11))
    #  print(np.any(results[2] - results2[2]))

    res2 = calculate_coordinates(model.domain_size,
                                 rec_pos=[center_pos],
                                 angle=angles,
                                 distance=results[0],
                                 amplitude=results[1])
    print(res2)

    print(model.domain_size)
    #  plt.xlim(0, model.domain_size[0])
    #  plt.ylim(0, model.domain_size[1])
    #  plt.gca().invert_yaxis()
    #  plt.scatter(res2[0, :, 0], res2[0, :, 1])
    #  plt.show()
    compare_velocity_to_measure(model,
                                res2[0],
                                source=src.coordinates.data,
                                receiver=rec.coordinates.data)
    x = results[2][0, :, 64]
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    first_peak = peaks[(prominences -
                        np.average(prominences)) > np.std(prominences)]
    distance = [((p * model.critical_dt) / 2) * v_water for p in first_peak]
    plt.plot(x)
    plt.plot(peaks, x[peaks], 'ro')
    plt.plot(first_peak, x[first_peak], 'bx')
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
