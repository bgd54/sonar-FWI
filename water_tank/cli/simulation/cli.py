import typer

import numpy as np

from simulation.sonar import Sonar
from simulation.utils import Bottom
from simulation.plotting import PlotType

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
    tn: float = typer.Argument("-t", help="End time of the simulation. (ms)"),
    bottom: Bottom = Bottom.ellipsis,
):
    """Initialize the sonar class and run the simulation."""
    s = Sonar(size_x, size_y, f0, v_env, tn, ns, posx, posy, bottom)
    s.run_position_angles(5, 10, 5)

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
    tn: float = typer.Argument("-t", help="End time of the simulation. (ms)"),
    bottom: Bottom = Bottom.ellipsis,
    start_angle: float = typer.Option(30., "-a", help="First angle for a beam."),
    last_angle: float = typer.Option(150., "-e", help="Last angle for a beam."),
    angle_step: float = typer.Option(1., "-s", help="Step size for angles"),
    output: str = typer.Option("./beams.npy", "-o", help="output file to save recordings"),
):
    """Initialize the sonar class."""
    s = Sonar(size_x, size_y, f0, v_env, tn, ns, posx, posy, bottom)
    angles = np.arange(start_angle, last_angle, angle_step)
    recordings = s.run_angles(angles)
    with open(output, 'wb') as fout:
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
    tn: float = typer.Argument("-t", help="End time of the simulation. (ms)"),
    bottom: Bottom = Bottom.ellipsis,
    plot_type: PlotType = PlotType.model,
):
    """Initialize the sonar class and plot the result."""
    s = Sonar(size_x, size_y, f0, v_env, tn, ns, posx, posy, bottom)
    s.plot_model (PlotType.model)


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
    tn: float = typer.Argument("-t", help="End time of the simulation. (ms)"),
    bottom: Bottom = Bottom.ellipsis,
    in_file: str = typer.Option("./beams.npy", "-i", help="input file to load recordings"),
):
    """Initialize the sonar class."""
    s = Sonar(size_x, size_y, f0, v_env, tn, ns, posx, posy, bottom)
    with open(in_file, 'rb') as fin:
        angles = np.load(fin)
        recordings= np.load(fin)
    s.parse_and_plot(angles, recordings)
    

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
