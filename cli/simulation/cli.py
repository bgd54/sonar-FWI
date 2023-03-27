import numpy as np
import typer
import math
import matplotlib.pyplot as plt

from simulation.plotting import PlotType, plot_snapshot_and_signal
from simulation.sonar import Sonar, Sonar_v2
from simulation.utils import Bottom, FlatBottom, CircleBottom, EllipsisBottom

app = typer.Typer()


@app.command()
def run(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.2, "-d", help="Distance between sources (m)"
    ),
    bottom: Bottom = Bottom.ellipsis,
    r: float = typer.Option(28.0, "-r", help="Radius of the bottom circle. (m)"),
    obstacle: bool = typer.Option(False, "-o"),
):
    """Initialize the sonar class and run the simulation."""
    s = Sonar(
        size_x,
        size_y,
        f0,
        v_env,
        ns,
        posx,
        posy,
        bottom,
        source_distance,
        obstacle=obstacle,
        r=r,
    )
    s.run_position_angles(5, 10, 5)
    plt.show()


@app.command()
def beams(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.2, "-d", help="Distance between sources (m)"
    ),
    bottom: Bottom = Bottom.ellipsis,
    r: float = typer.Option(28.0, "-r", help="Radius of the bottom circle. (m)"),
    obstacle: bool = typer.Option(False, "--obstacle"),
    start_angle: float = typer.Option(30.0, "-a", help="First angle for a beam."),
    last_angle: float = typer.Option(150.0, "-e", help="Last angle for a beam."),
    angle_step: float = typer.Option(1.0, "-s", help="Step size for angles"),
    output: str = typer.Option(
        "./beams.npy", "-o", help="output file to save recordings"
    ),
):
    """Initialize the sonar class."""
    s = Sonar(
        size_x,
        size_y,
        f0,
        v_env,
        ns,
        posx,
        posy,
        bottom,
        source_distance,
        obstacle=obstacle,
        r=r,
    )
    angles = np.arange(start_angle, last_angle, angle_step)
    recordings = s.run_angles(angles)
    with open(output, "wb") as fout:
        np.save(fout, angles)
        np.save(fout, recordings)


@app.command()
def plot(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.2, "-d", help="Distance between sources (m)"
    ),
    bottom: Bottom = Bottom.ellipsis,
    obstacle: bool = typer.Option(False, "-o"),
    plot_type: PlotType = PlotType.model,
):
    """Initialize the sonar class and plot the result."""
    s = Sonar(
        size_x,
        size_y,
        f0,
        v_env,
        ns,
        posx,
        posy,
        bottom,
        source_distance,
        obstacle=obstacle,
    )
    s.plot_model(plot_type)


@app.command()
def analyse(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.2, "-d", help="Distance between sources (m)"
    ),
    bottom: Bottom = Bottom.ellipsis,
    r: float = typer.Option(28.0, "-r", help="Radius of the bottom circle. (m)"),
    obstacle: bool = typer.Option(False, "--obstacle"),
    outfile: str = typer.Option(
        "./plot.png", "-o", help="Output file to save figure to."
    ),
    in_file: str = typer.Option(
        "./beams.npy", "-i", help="input file to load recordings"
    ),
):
    """Initialize the sonar class."""
    s = Sonar(
        size_x,
        size_y,
        f0,
        v_env,
        ns,
        posx,
        posy,
        bottom,
        source_distance,
        obstacle=obstacle,
        r=r,
    )
    with open(in_file, "rb") as fin:
        angles = np.load(fin)
        recordings = np.load(fin)
    s.parse_and_plot(angles, recordings)
    plt.savefig(outfile)


@app.command()
def snaps(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.2, "-d", help="Distance between sources (m)"
    ),
    bottom: Bottom = Bottom.ellipsis,
    obstacle: bool = typer.Option(False, "--obstacle"),
    alpha: float = typer.Option(80, "-a", help="Angle of the beam"),
    snaps_rate: float = typer.Option(0.1, "-s", help="Time between snapshots (ms)"),
    outfile: str = typer.Option(
        "./beam_tmp.mp4", "-o", help="Output file to save video to."
    ),
):
    """Initialize the sonar class."""
    s = Sonar(
        size_x,
        size_y,
        f0,
        v_env,
        ns,
        posx,
        posy,
        bottom,
        source_distance,
        snaps_rate,
        obstacle=obstacle,
    )
    s.run_angles(np.arange(alpha, alpha + 1))
    plot_snapshot_and_signal(s.usave.data, s.rec.data, s.model, outfile)


@app.command()
def beams_v2(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.02, "-d", help="Distance between sources (m)"
    ),
    start_angle: float = typer.Option(
        30, "-sa", help="Start angle of the beam. (degrees)"
    ),
    last_angle: float = typer.Option(
        150, "-la", help="Last angle of the beam. (degrees)"
    ),
    out_file: str = typer.Option(
        "./beams.npy", "-o", help="Output file to save recordings to."
    ),
):
    max_distance = math.sqrt((size_x / 2) ** 2 + (size_y) ** 2)
    t_end = max_distance / v_env
    s = Sonar_v2((size_x, size_y), f0, v_env, CircleBottom(posx, posy, 28), t_end)
    s.set_sine_source(ns, (size_x, size_y), source_distance)
    s.finalize()
    angles = np.arange(start_angle, last_angle, 5)
    recordings = s.run_angles(angles, source_distance, (posx, posy))
    with open(out_file, "wb") as fout:
        np.save(fout, angles)
        np.save(fout, recordings)


@app.command()
def analyse_v2(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(
        0.5, "-px", help="Position of the source in x direction. (relative)"
    ),
    posy: float = typer.Option(
        0.0, "-py", help="Position of the source in y direction. (relative)"
    ),
    source_distance: float = typer.Option(
        0.02, "-d", help="Distance between sources (m)"
    ),
    outfile: str = typer.Option(
        "./plot.png", "-o", help="Output file to save figure to."
    ),
    in_file: str = typer.Option(
        "./beams.npy", "-i", help="input file to load recordings"
    ),
):
    max_distance = math.sqrt((size_x / 2) ** 2 + (size_y) ** 2)
    t_end = max_distance / v_env
    s = Sonar_v2((size_x, size_y), f0, v_env, CircleBottom(posx, posy, 28), t_end)
    ideal_signal = s.get_ideal_signal()
    s.set_sine_source(ns, (size_x, size_y), source_distance)
    s.finalize()
    with open(in_file, "rb") as fin:
        angles = np.load(fin)
        recordings = np.load(fin)
    s.parse_and_plot(recordings, angles, ideal_signal)
    plt.savefig(outfile)


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
