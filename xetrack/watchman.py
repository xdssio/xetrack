import platform
import typer
import subprocess
import os
from typing import List
import json
import shelve
import logging
from xetrack.cli import run

GEPATH = "GEPATH"

logger = logging.getLogger(__name__)



class Watchman:
    LOCK = 'LOCK'

    def __init__(self, db='~/watchman.db', verbose: bool = False, ):
        self.db = db
        self.verbose = verbose

    @property
    def locked(self):
        return self.get(Watchman.LOCK) == 1

    def lock(self):
        """This is a primitive lock which is not very good"""
        if self.locked:
            return False
        self.set('LOCK', 1)
        return True

    def release(self):
        self.set('LOCK', 0)

    def get(self, key: str):
        with shelve.open(self.db, flag='xb') as db:
            return db[key]

    def set(self, key: str, value: str):
        with shelve.open(self.db) as db:
            db[key] = value

    @staticmethod
    def _to_string(patterns: List[str]):
        return json.dumps(patterns).replace('"', "'").replace(',', '')[1:-1]

    def watch(self, triggers: List[str], commit: List[str] = None, cwd: str = ''):
        """Watch the current directory for changes and commit them"""
        # pattern = json.dumps(triggers).replace('"',"'").replace(',','')[1:-1]
        pattern = self._to_string(triggers)
        commits = pattern if commit is None else self._to_string(commit)
        cwd = os.getcwd() if not cwd else cwd
        command = f"watchman-make -p {pattern} --run \"(cd {cwd} && git add {commits} && git commit -m 'auto-commit {pattern}->{commits}')\""
        if cwd:
            command = command[:-1] + f" --cwd={cwd}\""
        print(command)
        command = f"start /B {command}" if platform.system() == 'Windows' else f"nohup {command} &"  # background process

        run(command, cwd=cwd)

    @staticmethod
    def has_special(pattern: str):
        """Check if the pattern contains special characters"""
        return len(set(pattern).intersection({'*', '?', '[', ']', '!'})) > 0

    def ls(self, all: bool = typer.Option(False, '--all', help="List all watchman processes")):
        """List all watchman processes in the current directory"""

        valid_paths_count, path = 0, os.getcwd()
        for process in self._list_processes():
            if not all and path != process.path:
                continue
            message = f"watchman {process.service} for pattern '{process.pattern}' on pid {process.pid}"
            if process.path:
                message = message + f" in {process.path}"
                valid_paths_count += 1

        if valid_paths_count == 0:
            if not all:
                print(f"No watchman processes found for path {path} - consider using `--all`")
            else:
                print(f"No watchman processes found")

    def _list_processes(self):
        """List all watchman processes"""
        command = ' | '.join(["ps aux", f"grep 'watchman-make'"])
        output = run(command).stdout.decode("utf-8").strip().split('\n')

        processes = []
        for row in output:
            if 'grep' in row:
                continue
            result = row.split('xetrack')[-1]
            print(row)
            # row = row.split(' ')
            # pid = None
            # for i, s in enumerate(row):
            #
            #     if not pid and s.isdigit():
            #         pid = int(s)
            #     if 'watchman-make' in s:
            #         pattern, service = row[i + 2], row[i + 5]
            #         path = self.get_pid_path(pid)
            #         processes.append(pattern, service, pid, path)
            #         continue
        return processes

    @staticmethod
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
            print(f"Error getting environment for pid {pid}: {e}")

    @staticmethod
    def validate_pywatchman():
        """Check if pywatchman is installed"""
        try:
            import pywatchman
            return True
        except ImportError:
            return False

    @staticmethod
    def kill_pid(pid):
        """Kill a given pid"""
        response = subprocess.run(f"kill {pid}", shell=True, capture_output=True)
        if response.returncode != 0:
            print(f"{response.stderr.decode()}")
            return False
        else:
            print(f"Killed pid {pid}")
            return True
# #
# #
# # def stop(service: Annotated[str, typer.Option(help="Which service to type to stop")] = None,
# #          pattern: Annotated[str, typer.Option(help="Which pattern to - default all")] = None,
# #          pid: Annotated[int, typer.Option(help="A single pid to kill")] = None,
# #          all: Annotated[bool, typer.Option('--all', help="Kill all watchman processes")] = False):
# #     """Stop a watchman process"""
# #     killed = False
# #     if not all and pid:
# #         return kill_pid(pid)
# #     for process in _list_processes():
# #         if all:
# #             killed = killed or kill_pid(process.pid)
# #         elif service and process.service == service:
# #             killed = killed or kill_pid(process.pid)
# #         elif pattern and process.pattern == pattern:
# #             killed = killed or kill_pid(process.pid)
# #     if not killed:
# #         console.print(f"No watchman processes found", style=Style(color="red"))
#     elif all:
#         console.print(f"Killed all watchman processes", style=Style(color="green"))
