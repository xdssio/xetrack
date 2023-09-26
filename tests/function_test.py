from tempfile import NamedTemporaryFile

from xetrack import Tracker


def test_track_function():
    tmp = NamedTemporaryFile()
    tracker = Tracker(db=tmp.name, params={"model": 'lightgbm'}, reset=True)

    def foo(a: int, b: str):
        return a + len(b)

    assert tracker.track(foo, kwargs={'a': 1, 'b': 'hello'}) == 6
    assert tracker.track(foo, args=[1, 'hello']) == 6
    assert tracker.latest['function_name'] == 'foo'
    tracker.track(foo, params={'name': 'bar'}, args=[1, 'hello'])
    assert tracker.latest['function_result'] == 6
    assert tracker.latest['name'] == 'bar'


def test_wrapper():
    tmp = NamedTemporaryFile()
    tracker = Tracker(db=tmp.name, params={"model": 'lightgbm'}, reset=True)

    @tracker.wrap(params={'name': 'foofoo'})
    def foo(a: int, b: str):
        return a + len(b)

    result = foo(1, 'hello')

    assert tracker.latest[tracker.FUNCTION_RESULT] == result
    assert tracker.latest['name'] == 'foofoo'
