"""Setup pytest configuration, applied to scope of test folder"""


def pytest_addoption(parser):
    """Comma separated list of PMT backends to configure"""
    parser.addoption("--pmt", action="store", default="rapl")
