import numpy as np
import typer
import math
import matplotlib.pyplot as plt

from simulation.plotting import PlotType, plot_snapshot_and_signal
from simulation.sonar import Sonar
from simulation.utils import FlatBottom, EllipsisBottom, CircleBottom, run_beam

from devito.mpi import MPI
from devito import mode_performance, set_log_level

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
    mpi: bool = typer.Option(False, "-m", help="Run with MPI"),
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
    MPI.Init()
    set_log_level("DEBUG", comm=MPI.COMM_WORLD)
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
    MPI.Finalize()
    with open(output, "wb") as f:
        np.save(f, recording)


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
