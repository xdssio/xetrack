from xetrack import Tracker


class Mock():
    def __init__(self, name: str) -> None:
        self.name = name


def test_types():
    tracker = Tracker(Tracker.IN_MEMROY, logs_stdout=True)
    payload = {'str': 'str', 'int': 1, 'float': 1.0, 'bool': True, 'list': [
        1, 2, 3], 'dict': {'a': 1, 'b': 2}, 'class': Mock('test')}
    tracker.log(payload)
