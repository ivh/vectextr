import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--fits-file",
        action="store",
        default=None,
        help="Path to FITS file for extraction test",
    )
    parser.addoption(
        "--slitchar-file",
        action="store",
        default=None,
        help="Path to NPZ file containing slitdeltas (median_offsets)",
    )


@pytest.fixture
def fits_file(request):
    """Fixture to get FITS file path from command line."""
    return request.config.getoption("--fits-file")


@pytest.fixture
def slitchar_file(request):
    """Fixture to get slitchar NPZ file path from command line."""
    return request.config.getoption("--slitchar-file")
