# Sonar Simulation
This project aims to develop a simulational environment for exploring river beds using sonar signals.

## Usage
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
  run  Initialize the sonar class.
```

You can run the following command to access the `run` command's usage description:

```bash
$ python -m simulation run --help
Usage: sonar run [OPTIONS] [TN]

  Initialize the sonar class.

Arguments:
  [TN]  End time of the simulation. (ms)  [default: -t]

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
  --help      Show this message and exit.
```