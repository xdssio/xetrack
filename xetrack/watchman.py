import platform
import typer
import subprocess
import os
from gitease.annotations import message_annotation
from typing_extensions import Annotated
from collections import namedtuple
from enum import Enum
from rich.console import Console
from rich.style import Style

from gitease.git_helper import GitHelper

GEPATH = "GEPATH"
console = Console()


class ServiceEnum(str, Enum):
    SAVE = "save"
    SHARE = "share"


Process = namedtuple('Process', ['pattern', 'service', 'pid', 'path'])


def is_special(pattern: str):
    """Check if the pattern contains special characters"""
    return len(set(pattern).intersection({'*', '?', '[', ']', '!'})) > 0


def watch(pattern: str = typer.Argument(help="The pattern to watch for changes"),
          service: ServiceEnum = typer.Argument(help="Which automation to run"),
          message: message_annotation = None,
          detach: bool = typer.Option(False, '--detach', help="Detach and run in the background")):
    """Watch for changes to a files which match a pattern and run a 'save' or 'share' when changes are detected"""
    console.print(f"Watching for changes to {pattern} in and running {service}",
                  style=Style(color="yellow", blink=True, bold=True))
    if not validate_pywatchman:
        console.print("Install pywatchman to use this feature: 'pip install pywatchman'",
                      style=Style(color="red", blink=True, bold=True))
        return False
    helper = GitHelper()
    command = f"watchman-make -p '{pattern}' --run 'ge {service} -y -a \"{pattern}\""
    if message:
        command = command + f" -m {message}"
    command = command + "'"
    if detach:
        command = f"start /B {command}" if platform.system() == 'Windows' else f"nohup {command} &"  # background process
    console.print(f"Running command: {command}", style=Style(color="yellow", blink=True, bold=True))
    command = f"GEPATH={helper.repo.working_dir} {command}"
    subprocess.run(command, shell=True)
    if not detach:
        console.print(f"Stopped....", style=Style(color="red", blink=True, bold=True))


def ls(all: bool = typer.Option(False, '--all', help="List all watchman processes")):
    """List all watchman processes in the current directory"""

    valid_paths_count, path = 0, os.getcwd()
    for process in _list_processes():
        if not all and path != process.path:
            continue
        message = f"watchman {process.service} for pattern '{process.pattern}' on pid {process.pid}"
        if process.path:
            message = message + f" in {process.path}"
            valid_paths_count += 1
        console.print(message, style=Style(color="yellow"))
    if valid_paths_count == 0:
        if not all:
            console.print(f"No watchman processes found for path {path} - consider using `--all`",
                          style=Style(color="red"))
        else:
            console.print(f"No watchman processes found", style=Style(color="green"))


def _list_processes():
    """List all watchman processes"""
    command = ' | '.join(["ps aux", f"grep 'watchman-make'"])
    output = subprocess.check_output(command, shell=True).decode("utf-8").strip().split('\n')
    processes = []
    for row in output:
        if 'grep' in row:
            continue
        row = row.split(' ')
        pid = None
        for i, s in enumerate(row):
            if not pid and s.isdigit():
                pid = int(s)
            if 'watchman-make' in s:
                pattern, service = row[i + 2], row[i + 5]
                path = get_pid_path(pid)
                processes.append(Process(pattern, service, pid, path))
                continue

    return processes


def get_pid_path(pid):
    """retrieve the path for a given pid using the GEPATH environment variable"""
    try:  # TODO: make this work on windows + tests

        if platform.system() == 'Windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            buf_size = kernel32.GetEnvironmentVariableW(GEPATH, None, 0, pid)
            if buf_size == 0:
                return None
            buf = ctypes.create_unicode_buffer(buf_size)
            kernel32.GetEnvironmentVariableW(GEPATH, buf, buf_size, pid)
            return buf.value
        else:
            result = subprocess.run(f"ps eww {pid}", shell=True, capture_output=True).stdout
            if result:
                for s in result.decode("utf-8").split(" "):
                    if GEPATH in s:
                        return s.split("=")[1]
    except Exception as e:
        console.print(f"Error getting environment for pid {pid}: {e}", style=Style(color="red"))


def validate_pywatchman():
    """Check if pywatchman is installed"""
    try:
        import pywatchman
        return True
    except ImportError:
        return False


def kill_pid(pid):
    """Kill a given pid"""
    response = subprocess.run(f"kill {pid}", shell=True, capture_output=True)
    if response.returncode != 0:
        console.print(f"{response.stderr.decode()}", style=Style(color="red"))
        return False
    else:
        console.print(f"Killed pid {pid}", style=Style(color="yellow"))
        return True


def stop(service: Annotated[str, typer.Option(help="Which service to type to stop")] = None,
         pattern: Annotated[str, typer.Option(help="Which pattern to - default all")] = None,
         pid: Annotated[int, typer.Option(help="A single pid to kill")] = None,
         all: Annotated[bool, typer.Option('--all', help="Kill all watchman processes")] = False):
    """Stop a watchman process"""
    killed = False
    if not all and pid:
        return kill_pid(pid)
    for process in _list_processes():
        if all:
            killed = killed or kill_pid(process.pid)
        elif service and process.service == service:
            killed = killed or kill_pid(process.pid)
        elif pattern and process.pattern == pattern:
            killed = killed or kill_pid(process.pid)
    if not killed:
        console.print(f"No watchman processes found", style=Style(color="red"))
    elif all:
        console.print(f"Killed all watchman processes", style=Style(color="green"))
