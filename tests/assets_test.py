import sys
import os
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from xetrack.assets import Assets


def generate_large_file(filename, size_in_mb):
    """ Generates a large file with random contents. """
    with open(filename, 'wb') as f:
        f.write(os.urandom(size_in_mb * 1024 * 1024))


@dataclass
class Content():
    text: str

    def __repr__(self):
        return f"Content(text={self.text[:20]}...)"


def test_large_file_deduplication():
    tmp = TemporaryDirectory()
    filename = f'{tmp.name}/large_test_file.dat'
    size_in_mb = 10  # Size of the file in megabytes
    generate_large_file(filename, size_in_mb)
    manager = Assets(f'{tmp.name}/db.db')

    file_size = sys.getsizeof(f"{tmp.name}/large_test_file.dat")
    db_size = sys.getsizeof(f"{tmp.name}/db.db")

    with open(filename, 'rb') as f:
        content = f.read()
    c = Content(content)  # type: ignore
    names = ['file1', 'file2', 'file3']
    for name in names:
        manager.insert(name, c)

    retrieved_files = [manager.get(name) for name in names]
    all_files_identical = all(
        c == retrieved_file for retrieved_file in retrieved_files)

    assert all_files_identical, "Test failed: Files retrieved are not identical"
