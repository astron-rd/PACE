def pytest_collection_modifyitems(items):
    for item in items:
        name = item.nodeid.split("::")[-1].removeprefix("test_")
        item._nodeid = f"idg_python_{name}"
