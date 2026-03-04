import pytest
from xetrack._dataframe import set_backend, PANDAS, POLARS


@pytest.fixture(params=[PANDAS, POLARS])
def df_backend(request):
    """Parametrized fixture that runs tests under both pandas and polars backends."""
    set_backend(request.param)
    yield request.param
    set_backend("auto")


@pytest.fixture(autouse=True)
def _ensure_pandas_default():
    """Reset backend to pandas before every test to protect tests that use raw pandas API."""
    set_backend(PANDAS)
    yield
    set_backend("auto")
