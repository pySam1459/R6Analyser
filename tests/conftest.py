import pytest

@pytest.fixture
def mock_config_args():
    """Fixture to provide mock command line arguments."""
    class MockArgs:
        def __init__(self):
            self.region_tool = False
    return MockArgs()
