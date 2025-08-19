from xetrack.git import get_commit_hash, get_commit_tags


def test_get_commit():
    commit_hash = get_commit_hash()
    assert isinstance(commit_hash, str)
    assert len(commit_hash) == 40

    tests_commit_hash = get_commit_hash('tests')
    # Note: commit_hash might equal tests_commit_hash if the latest commit modified both root and tests
    assert isinstance(tests_commit_hash, str)
    assert len(tests_commit_hash) == 40

    assert get_commit_hash(git_root='blabla') is None


def test_get_commit_tags():
    assert isinstance(get_commit_tags(), list)
