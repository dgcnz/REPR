import pytest
import tabulate
from .runner import benchmark_v2

def pytest_terminal_summary(terminalreporter, exitstatus, config):

    passed = terminalreporter.stats.get("passed", [])
    names = [{"test": x.head_line} for x in passed]
    user_properties = [ dict(x.user_properties) for x in passed ]
    table = [ name | user_property for name, user_property in zip(names, user_properties) ]
    print(tabulate.tabulate(table, headers="keys"))

