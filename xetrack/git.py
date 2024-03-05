import subprocess
from typing import Optional, List
import os


def get_commit_hash(path: Optional[str] = None, git_root: str = '.') -> Optional[str]:
    """
    Get the commit hash of the latest commit in the Git repository.

    Args:
        path (Optional[str]): The path to the file for which to get the commit hash. If not provided, the commit hash of the repository will be returned. Default is None.
        git_root (str): The root directory of the Git repository. Default is the current directory.

    Returns:
        Optional[str]: The commit hash of the latest commit in the Git repository, or None if an error occurred.
    """

    try:
        if path:
            command = ['git', 'log', '-1', '--', os.path.join(git_root, path)]
            output = subprocess.check_output(command, cwd=git_root).decode()
            return output.split('\n')[0][len('commit '):]

        output = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=git_root)
        return output.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_commit_tags(git_root: str = '.') -> List[str]:
    """
    Get the current Git tags assosiacted with the repository.

    Args:
        git_root (str): The root directory of the Git repository. Default is the current directory.

    Returns:
        List[str]: A list of the current Git tags in the repository.
    """
    try:
        command = ['git', 'tag', '--poi']
        output = subprocess.check_output(command, cwd=git_root).decode()
        tags = output.split('\n')
        if '' in tags:
            tags.remove('')
        return tags
    except subprocess.CalledProcessError:
        return []
