from typing import Annotated

import typer
import logging
import subprocess

cli = typer.Typer(add_completion=True)
logger = logging.getLogger(__name__)


def run(command: str, cwd: str = ''):
    logger.debug(command)
    args = {"shell": True, "capture_output": True}
    if cwd:
        args["cwd"] = cwd
    out = subprocess.run(command, **args)
    logger.debug(out)
    return out


@cli.command()
def commit(patterns: Annotated[str, typer.Argument(help="Patterns to add and commit")] = '.',
           cwd: Annotated[str, typer.Option(help="Working directory")] = ''):
    """Commit changes"""
    command = f"""
    git add {patterns}
    git commit -m "Watchman commit for patterns {patterns}"
    git push
    """

    print(command)
    print(run(command, cwd=cwd))


@cli.command()
def kill(pid: Annotated[int, typer.Argument(help="Process ID")]):
    """Kill a process"""
    print(pid)
