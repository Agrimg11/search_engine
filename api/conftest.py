import pytest

# Configure pytest-asyncio to auto mode so all async tests/fixtures work
# without needing explicit @pytest.mark.asyncio decorators on each test.
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as asyncio"
    )
