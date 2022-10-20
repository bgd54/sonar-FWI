import typer

from simulation.sonar import Sonar
from simulation.utils import Bottom

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
    """Initialize the sonar class."""
    s = Sonar(size_x, size_y, f0, v_env, tn, ns, posx, posy, bottom)
    s.run_position_angles(5, 10, 5)


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
