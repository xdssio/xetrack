import sys
import os
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from xetrack.assets import AssetsManager


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
    manager = AssetsManager(f'{tmp.name}/db.db')

    file_size = sys.getsizeof(f"{tmp.name}/large_test_file.dat")
    db_size = sys.getsizeof(f"{tmp.name}/db.db")

    with open(filename, 'rb') as f:
        content = f.read()
    content = Content(text=content)  # type: ignore
    names = ['file1', 'file2', 'file3']
    hash_id = ''
    for name in names:
        hash_id = manager.insert(name, content)

    assert manager.get('file1')
    assert manager.get(hash_id)

    assert all(
        content == retrieved_file for retrieved_file in [manager.get(name) for name in names])

    assert hash_id != manager.insert('file2', Content('file2'))

    assert any(
        content != retrieved_file
        for retrieved_file in [manager.get(name) for name in names]
    )
