import pytest
from xetrack._dataframe import set_backend, PANDAS, POLARS


@pytest.fixture(params=[PANDAS, POLARS])
def df_backend(request):
    """Parametrized fixture that runs tests under both pandas and polars backends."""
    set_backend(request.param)
    yield request.param
    set_backend("auto")
