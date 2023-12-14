from tempfile import NamedTemporaryFile

from xetrack import Tracker, TRACKER_CONSTANTS


def test_track_function():
    tracker = Tracker(db=Tracker.IN_MEMROY, params={"model": 'lightgbm'})

    def foo(a: int, b: str):
        return a + len(b)

    assert tracker.track(foo, kwargs={'a': 1, 'b': 'hello'}) == 6
    assert tracker.track(foo, args=[1, 'hello']) == 6
    assert tracker.latest['function_name'] == 'foo'  # type: ignore
    assert tracker.track(foo, params={'name': 'bar'}, args=[1, 'hello']) == 6
    assert tracker.latest['function_result'] == 6  # type: ignore
    assert tracker.latest['name'] == 'bar'  # type: ignore


def test_wrapper():

    tracker = Tracker(db=Tracker.IN_MEMROY, params={"model": 'lightgbm'})

    @tracker.wrap(params={'name': 'foofoo'})
    def foo(a: int, b: str):
        return a + len(b)

    result = foo(1, 'hello')

    # type: ignore
    assert tracker.latest[TRACKER_CONSTANTS.FUNCTION_RESULT] == result
    assert tracker.latest['name'] == 'foofoo'  # type: ignore
