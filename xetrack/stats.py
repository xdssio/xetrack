import typing
import time
import multiprocessing
import multiprocessing.managers
import psutil


class Stats(object):
    KEYS = ['disk_percent', 'p_memory_percent', 'cpu', 'memory_percent']

    def __init__(self, process: psutil.Process, stats: multiprocessing.managers.DictProxy, interval: float = 0.1):
        self.interval = interval
        self.process = process
        self.count = 0
        self.stats = stats
        for key in Stats.KEYS:
            self.stats[key] = 0

    @staticmethod
    def round10e5(val):
        return round(val * 10e5) / 10e5

    def get_stats(self):
        memory_usage = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        return {
            # CPU utilization percent(can be over 100%)
            'cpu': self.round10e5(self.process.cpu_percent(0.0)),

            # Whole system memory usage
            # 'memory_used': round10e5(memory_usage.used / 1024 / 1024),
            'memory_percent': self.round10e5(memory_usage.used * 100 / memory_usage.total),

            # Get the portion of memory occupied by a process
            # 'p_memory_rss': round10e5(self._process.memory_info().rss
            #                           / 1024 / 1024),
            'p_memory_percent': self.round10e5(self.process.memory_percent()),

            # Disk usage
            # 'disk_used': round10e5(disk_usage.used / 1024 / 1024),
            'disk_percent': self.round10e5(disk_usage.percent),
        }

    def get_average_stats(self) -> dict:
        if self.count == 0:
            return self.stats
        for key, value in self.stats.items():
            self.stats[key] = value / self.count

        return self.stats

    def collect_stats(self, stop_event: multiprocessing.Event):
        while not stop_event.is_set():
            self.count += 1
            for key, value in self.get_stats().items():
                self.stats[key] += value
            time.sleep(self.interval)
