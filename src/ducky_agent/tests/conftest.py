import pytest


@pytest.fixture
def anyio_backend():
    # pytest-asyncio isn't installed in this environment; anyio's own
    # pytest plugin (already present as a transitive textual dependency)
    # covers the same "await an async test function" need without adding
    # a new dependency. asyncio only -- trio isn't installed either, and
    # Textual itself runs on asyncio.
    return "asyncio"
