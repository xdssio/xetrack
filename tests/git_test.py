from xetrack.git import get_commit_hash, get_commit_tags


def test_get_commit():
    commit_hash = get_commit_hash()
    assert isinstance(commit_hash, str)
    assert len(commit_hash) == 40

    tests_commit_hash = get_commit_hash('tests')
    assert commit_hash != tests_commit_hash

    assert get_commit_hash(git_root='blabla') is None


def test_get_commit_tags():
    assert isinstance(get_commit_tags(), list)
