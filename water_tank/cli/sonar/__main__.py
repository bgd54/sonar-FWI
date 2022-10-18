"""Entry point of the sonar command line interface."""

from sonar import cli

def main():
    cli.app(prog_name="sonar")

if __name__ == "__main__":
    main()