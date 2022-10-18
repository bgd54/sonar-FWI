import typer

from sonar import sonar
import numpy as np
app = typer.Typer()

@app.command()
def run(
    size_x: int = typer.Option(60, "-x", help="Size in x direction. (m)"),
    size_y: int = typer.Option(30, "-y", help="Size in y direction. (m)"),
    f0: float = typer.Option(5, "-f", help="Center frequency of the signal. (kHz)"),
    v_env: float = typer.Option(1.5, "-v", help="Environment velocity. (km/s)"),
    tn: float = typer.Option(50, "-t", help="End time of the simulation. (ms)"),
    ns: int = typer.Option(128, "-n", help="Number of sources."),
    posx: float = typer.Option(.5, "-px", help="Position of the source in x direction. (relative)"),
    posy: float = typer.Option(.5, "-py", help="Position of the source in y direction. (relative)"),
):
    """Initialize the sonar class."""
    angles = np.arange(5, 175, 5)
    sonar.Sonar(size_x, size_y, f0, v_env).run(tn, ns, posx, posy, angles)


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