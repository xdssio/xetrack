from tempfile import NamedTemporaryFile

from xetrack import Tracker


def test_track_function():
    tmp = NamedTemporaryFile()
    tracker = Tracker(db=tmp.name, params={"model": 'lightgbm'}, reset=True)

    def foo(a: int, b: str):
        return a + len(b)

    assert tracker.track(foo, kwargs={'a': 1, 'b': 'hello'})['function_result'] == 6
    assert tracker.track(foo, args=[1, 'hello'])['function_result'] == 6
    assert tracker.track(foo, args=[1, 'hello'])['name'] == 'foo'
    assert tracker.track(foo, name='bar', args=[1, 'hello'])['name'] == 'bar'
    data = tracker.track(foo, name='bar', params={'a': 2}, args=[1, 'hello'])
    assert data['function_result'] == 6
    assert data['a'] == 2


def test_wrapper():
    tmp = NamedTemporaryFile()
    tracker = Tracker(db=tmp.name, params={"model": 'lightgbm'}, reset=True)

    @tracker.wrap(params={'name': 'foofoo'})
    def foo(a: int, b: str):
        return a + len(b)

    result = foo(1, 'hello')

    assert tracker.latest[tracker.FUNCTION_RESULT] == result
    assert tracker.latest['name'] == 'foofoo'
