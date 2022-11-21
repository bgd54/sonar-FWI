# sonar-FWI

This project aims to develop a simulation environment for exploring river beds using sonar signals. Our work focuses on developing a parallel implementation in [Devito](https://www.devitoproject.org/) that can simulate the propagation and detection of the reflected signals.

## CLI

You can run the following command to access the application's usage description:

```bash
$ python -m simulation --help 
Usage: sonar [OPTIONS] COMMAND [ARGS]...

  Sonar: a Python package for sonar signal processing.

Options:
  -v, --verbose                   Show debug messages.
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

Commands:
  plot     Display different plots.
  beams    Run the simulation in different beam angles.
  analyse  Takes the angles and recordings and creates a plot.
  snaps    Create an animation of the snapshots.
  run      Initialize the sonar class.
```

You can run the following command to access the `run` command's usage description:

```bash
$ python -m simulation run --help
Usage: sonar run [OPTIONS]

  Initialize the sonar class and run the simulation.

Options:
  -x INTEGER  Size in x direction. (m)  [default: 60]
  -y INTEGER  Size in y direction. (m)  [default: 30]
  -f FLOAT    Center frequency of the signal. (kHz)  [default: 5]
  -v FLOAT    Environment velocity. (km/s)  [default: 1.5]
  -n INTEGER  Number of sources.  [default: 128]
  -px FLOAT   Position of the source in x direction. (relative)  [default:
              0.5]
  -py FLOAT   Position of the source in y direction. (relative)  [default:
              0.5]
  -d  FLOAT   Distance between sources (m) [default: 0.2]
  --bottom [flat|ellipsis|circle] [default: Bottom.ellipsis]
  -r FLOAT    Radius of the bottom circle. (m) [default: 28.0]
  -o          Obstacle flag. [default: False]
  --help      Show this message and exit.
```

## Notebooks
A series of Jupyter notebooks of incremental complexity are available in the `notebooks` folder. They are intended to show how the simulation works and how to use the different functions.