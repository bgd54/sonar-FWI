from pathlib import Path

import typer
import numpy as np
import matplotlib.pyplot as plt

from typing_extensions import Annotated, List
from simulation.sonar import Sonar
from simulation.plotting import plot_velocity
from simulation.utils import (
    EllipsisBottom,
    CircleBottom,
    rec2coords,
)

app = typer.Typer()


@app.command()
def run_single_freq_circ(
    angles: List[int] = typer.Argument(help="Angle of the beam (deg)"),
    x: Annotated[int, typer.Option(help="Size in x direction. (m)")] = 60,
    y: Annotated[int, typer.Option(help="Size in y direction. (m)")] = 30,
    f0: Annotated[
        float, typer.Option("-f", help="Center frequency of the signal. (kHz)")
    ] = 50,
    v_env: Annotated[float, typer.Option(help="Environment velocity. (km/s)")] = 1.5,
    ns: Annotated[int, typer.Option(help="Number of sources.")] = 128,
    sd: Annotated[float, typer.Option(help="Distance between sources (m)")] = 0.002,
    r: Annotated[float, typer.Option(help="Radius of the circle (m)")] = 0.5,
    plot: Annotated[bool, typer.Option(help="Plot the results")] = False,
    dir: Annotated[str, typer.Option(help="Folder to save the plots")] = "./",
):
    """Initialize the sonar class and run the simulation with 1 frequency."""
    cy = (ns - 1) / 2 * sd + sd
    sonar = Sonar(
        (x, y),
        f0,
        v_env,
        CircleBottom(x / 2, cy, r),
        source_distance=sd,
        ns=ns,
    )
    sonar.set_source()
    sonar.set_receiver()
    sonar.finalize()

    sonar.save_ideal_signal(dir + f"/circle/{f0}/ideal_signal")

    for a in angles:
        sonar.run_beam(a)
        sonar.save_recording(dir + f"/circle/{f0}/recording_{a}")


@app.command()
def run_single_freq_ellipsis(
    angles: List[int] = typer.Argument(help="Angle of the beam (deg)"),
    x: Annotated[int, typer.Option(help="Size in x direction. (m)")] = 60,
    y: Annotated[int, typer.Option(help="Size in y direction. (m)")] = 30,
    f0: Annotated[
        int, typer.Option("-f", help="Center frequency of the signal. (kHz)")
    ] = 50,
    v_env: Annotated[float, typer.Option(help="Environment velocity. (km/s)")] = 1.5,
    ns: Annotated[int, typer.Option(help="Number of sources.")] = 128,
    sd: Annotated[float, typer.Option(help="Distance between sources (m)")] = 0.002,
    dir: Annotated[str, typer.Option(help="Folder to save the plots")] = "./",
    plot: Annotated[bool, typer.Option(help="Plot the results")] = False,
):
    sonar = Sonar(
        (x, y),
        f0,
        v_env,
        EllipsisBottom(True),
        source_distance=sd,
        ns=ns,
    )
    sonar.set_source()
    sonar.set_receiver()
    sonar.finalize()

    if dir[-1] == "/":
        dir = dir[:-1]
    Path(f"./output/ellipsis/{f0}").mkdir(parents=True, exist_ok=True)

    sonar.save_ideal_signal(dir + f"/ellipsis/{f0}/ideal")

    recordings = {}
    for a in angles:
        sonar.run_beam(a)
        recordings[a] = sonar.recording
        sonar.save_recording(dir + f"/ellipsis/{f0}/recording_{a}")

    if plot:
        if sonar.model.grid.distributor.nprocs > 1:
            rank = sonar.model.grid.distributor.myrank
            comm = sonar.model.grid.distributor.comm
            dt = comm.gather(sonar.model.critical_dt, root=0)
            rec_coords = comm.gather(sonar.rec.coordinates.data, root=0)
            if rank == 0:
                coords = np.zeros((np.size(angles), 2))
                for i, (a, r) in enumerate(recordings.items()):
                    coords[i, :] = rec2coords(
                        r, rec_coords, sonar.src.signal_packet, a, dt
                    )
                else:
                    coords = None
            coords = comm.bcast(coords, root=0)
            plot_velocity(
                sonar.model,
                sonar.src.coordinates.data,
                sonar.rec.coordinates.data,
                coords,
                outfile=dir + f"/ellipsis/{f0}/velocity.pdf",
            )
        else:
            dt = sonar.model.critical_dt
            coords = np.zeros((np.size(angles), 2))
            for i, (a, r) in enumerate(recordings.items()):
                coords[i, :] = rec2coords(
                    r, sonar.rec.coordinates.data, sonar.src.signal_packet, a, dt
                )
            print(coords)
            plot_velocity(
                sonar.model,
                sonar.src.coordinates.data,
                sonar.rec.coordinates.data,
                coords,
                outfile=dir + f"/ellipsis/{f0}/velocity.pdf",
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
