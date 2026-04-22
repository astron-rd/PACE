def pytest_collection_modifyitems(items):
    for item in items:
        item._nodeid = item.nodeid.split("::")[-1].removeprefix("test_")
