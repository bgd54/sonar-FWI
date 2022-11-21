from cgi import test
from simulation.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_plot():
    command_name = "plot"
    args = [command_name, "50"]
    result = runner.invoke(app, args)


test_plot()
