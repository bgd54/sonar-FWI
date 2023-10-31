import numpy as np
import typer
import tqdm
import math
import matplotlib.pyplot as plt
from mpi4py import MPI

from simulation.plotting import (
    PlotType,
    plot_snapshot_and_signal,
    compare_velocity_to_measure,
)
from simulation.sonar import Sonar
from simulation.utils import (
    FlatBottom,
    EllipsisBottom,
    CircleBottom,
    run_beam,
)

app = typer.Typer()


@app.command()
def run_single_freq_circ(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    source_distance: float = typer.Option(
        0.002, "-d", help="Distance between sources (m)"
    ),
    alpha: float = typer.Option(0, "-a", help="Angle of the beam (deg)"),
    radius: float = typer.Option(28, "-r", help="Radius of the circle (m)"),
    output: str = typer.Option(
        "./recorded_signal.npy", "-o", help="output file to save recorded signal"
    ),
):
    """Initialize the sonar class and run the simulation with 1 frequency."""
    cy = (ns - 1) / 2 * source_distance + source_distance
    sonar = Sonar(
        (size_x, size_y),
        f0,
        v_env,
        CircleBottom(size_x / 2, cy, radius),
        source_distance=source_distance,
        ns=ns,
    )
    sonar.set_source()
    sonar.finalize()
    recording = run_beam(
        sonar.src,
        sonar.rec,
        sonar.op,
        sonar.u,
        source_distance,
        sonar.time_range,
        sonar.model.critical_dt,
        alpha,
        v_env,
    )
    with open(output, "wb") as f:
        np.save(f, recording)


@app.command()
def run_single_freq_ellipse(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    source_distance: float = typer.Option(
        0.002, "-d", help="Distance between sources (m)"
    ),
    alpha: float = typer.Option(0, "-a", help="Angle of the beam (deg)"),
    output: str = typer.Option(
        "./recorded_signal.npy", "-o", help="output file to save recorded signal"
    ),
    mpi: bool = typer.Option(
        False, "-m", help="Change the output saving when OpenMPI is used"
    ),
):
    """Initialize the sonar class and run the simulation with 1 frequency."""
    sonar = Sonar(
        (size_x, size_y),
        f0,
        v_env,
        EllipsisBottom(True),
        source_distance=source_distance,
        ns=ns,
    )
    sonar.set_source()
    sonar.finalize()

    recording = run_beam(
        sonar.src,
        sonar.rec,
        sonar.op,
        sonar.u,
        source_distance,
        sonar.time_range,
        sonar.model.critical_dt,
        alpha,
        v_env,
    )
    if mpi:
        rank = MPI.COMM_WORLD.Get_rank()
        all_recording = MPI.COMM_WORLD.gather(recording, root=0)
        if rank == 0:
            all_recording = np.concatenate(all_recording, axis=1)
            with open(output, "wb") as f:
                np.save(f, all_recording)
    else:
        with open(output, "wb") as f:
            np.save(f, all_recording)


@app.command()
def sonar_picture():
    domain_size = (60, 30)
    v_env = 1.5
    ns = 128
    source_distance = 0.002
    f0 = 100
    space_order = 8
    spatial_dist = round(v_env / f0 / 3, 3)
    dt = spatial_dist / 20
    angles = [30, 45, 60, 75, 90, 105, 120, 135, 150]
    obstacle = True
    v_wall = 5.64
    v_obj = 3.24
    domain_dims = (
        round(domain_size[0] / spatial_dist),
        round(domain_size[1] / spatial_dist),
    )
    vp = np.full(domain_dims, v_env, dtype=np.float32)
    r_obs = vp.shape[0] / 20
    a, b = vp.shape[0] / 4, vp.shape[1] - r_obs
    y, x = np.ogrid[-a : vp.shape[0] - a, -b : vp.shape[1] - b]
    vp[x * x + y * y <= r_obs * r_obs] = v_obj
    nx = domain_dims[0]
    nz = domain_dims[1]
    wall = round(nx * 0.02)
    offs = round(wall / 2)
    a = round((nx - wall) / 2)
    b = round((nz - wall) / 2)
    offs = round(wall / 2)
    x = np.arange(0, vp.shape[0])
    y = np.arange(0, vp.shape[1])
    if obstacle:
        r = vp.shape[0] / 100
        ox = np.arange(offs, 2 * a + offs + 1, 2 * a / 50)
        oy = np.sqrt(1 - (ox - a - offs) ** 2 / a**2) * b + offs + b
        for oxx, oyy in tqdm.tqdm(zip(ox, oy)):
            mask = (y[np.newaxis, :] - oyy) ** 2 + (
                x[:, np.newaxis] - oxx
            ) ** 2 < r**2
            vp[mask] = v_wall
    mask = (y[np.newaxis, :] - offs - b) ** 2 / b**2 + (
        x[:, np.newaxis] - offs - a
    ) ** 2 / a**2 > 1
    vp[mask] = v_wall
    vp[offs:-offs, :b] = v_env
    sonars = {
        a: Sonar(
            domain_size,
            f0,
            v_env,
            vp,
            space_order=space_order,
            dt=dt,
            spatial_dist=spatial_dist,
        )
        for a in angles
    }
    for _, v in sonars.items():
        v.set_source()
        v.finalize()
    ideal_signal = sonars[45].src.signal_packet
    recordings = {
        a: run_beam(
            sonars[a].src,
            sonars[a].rec,
            sonars[a].op,
            sonars[a].u,
            sonars[a].source_distance,
            sonars[a].time_range,
            sonars[a].model.critical_dt,
            a,
            v_env,
        )
        for a in angles
    }

    cords = np.zeros((np.size(angles), 2))
    for a, v in recordings.items():
        coordinates = np.zeros((128, 2))
        for i in range(128):
            start_time = np.argmax(recordings[a][:5000, i])
            correlate = np.correlate(recordings[a][5000:, i], ideal_signal, mode="same")
            peak = 5000 + correlate.argmax()
            distance = (peak - start_time) * sonars[a].model.critical_dt * v_env / 2
            rec_coords = sonars[a].rec.coordinates.data[i]
            coordinates[i, 0] = rec_coords[0] - distance * np.cos(np.deg2rad(a))
            coordinates[i, 1] = rec_coords[1] + distance * np.sin(np.deg2rad(a))
            cords[a // 15 - 2, :] = np.mean(coordinates, axis=0)
    compare_velocity_to_measure(
        sonars[45].model,
        cords,
        sonars[45].src.coordinates.data,
        sonars[45].rec.coordinates.data,
    )


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show debug messages.",
        show_default=False,
    )
):
    """Sonar: a Python package for sonar signal processing."""
    # utils.set_log_level(verbose)
