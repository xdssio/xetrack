from xetrack import Tracker
from tempfile import NamedTemporaryFile


def test_valid_keys():
    tmp = NamedTemporaryFile()
    tracker = Tracker(tmp.name, reset=True)
    data = tracker.log(**{"invalid key":1, '@@': 2, 'valid_key': 3})
    keys = set(data.keys())
    assert 'invalid key' not in keys
    assert 'invalid_key' in keys
    assert '@@' not in keys
    assert '__' in keys
    assert 'valid_key' in keys