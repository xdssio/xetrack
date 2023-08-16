from tempfile import TemporaryDirectory
from xetrack import Tracker
from functools import lru_cache
import pandas as pd
import numpy as np
import os
import sys
import requests
import pytest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

large_text_path = 'large_text.txt'


def _fibonacci_recursive(n):
    """Not efficient"""
    if n <= 1:
        return n
    else:
        return _fibonacci_recursive(n - 1) + _fibonacci_recursive(n - 2)


def fibonacci_recursive(n):
    return {'result': _fibonacci_recursive(n), 'n': n}


@lru_cache(maxsize=None)
def _fibonacci_memoization(n):
    if n <= 1:
        return n
    else:
        return _fibonacci_memoization(n - 1) + _fibonacci_memoization(n - 2)


def fibonacci_memoization(n):
    return {'result': _fibonacci_memoization(n), 'n': n}


def _fibonacci_tabulation(n):
    if n <= 1:
        return n
    else:
        table = [0, 1]
        for i in range(2, n + 1):
            table.append(table[i - 1] + table[i - 2])
        return table[n]


def fibonacci_tabulation(n):
    return {'result': _fibonacci_tabulation(n), 'n': n}


def memory_intensive_function(n: int):
    data = [0] * n  # 800 MB
    return {'result': n}


def cpu_intensive_function(n: int):
    data = [0] * n

    # Use a lot of CPU cycles
    for i in range(n):
        data[i] += 1

    return {'result': np.mean(data)}


def ram_intensive_function():
    import pyxet
    data = pd.read_csv('xet://xdssio/blog_corpus/main/data/data.csv')
    return {'result': len(data)}


def space_intensive_function(n: int):
    with open(large_text_path, 'w') as file:
        for _ in range(n):
            file.write('Lorem ipsum dolor sit amet\n')  # Write a large amount of text to the file

    return {'result': n}


def delete_large_text():
    if os.path.exists(large_text_path):
        os.remove(large_text_path)
        return {'result': 1}
    return {'result': 0}


def read_image():
    ret = requests.get(
        'https://images.unsplash.com/photo-1608501078713-8e445a709b39?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80')
    return {'result': sys.getsizeof(ret.content)}


@pytest.mark.skip(reason="Running locally")
def test_system_tracking():
    tempdir = TemporaryDirectory()
    database = tempdir.name + '/database.db'
    print(database)
    tracker = Tracker(database, log_system_params=True, log_network_params=True, verbose=True)

    n = 10000000
    tracker.track_function(cpu_intensive_function, n)
    tracker.track_function(memory_intensive_function, n)
    tracker.track_function(ram_intensive_function)
    tracker.track_function(space_intensive_function, n)
    tracker.track_function(delete_large_text)

    n = 35
    tracker.track_function(fibonacci_recursive, n)
    tracker.track_function(fibonacci_memoization, n)
    tracker.track_function(fibonacci_tabulation, n)

    tracker.track_function(read_image)
    print(tracker.to_df()[['name', 'time', 'memory_percent', 'cpu', 'disk_percent', 'p_memory_percent']])
