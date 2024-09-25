import io
import sys
import pytest


@pytest.fixture
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = original_stdout
