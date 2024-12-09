import pytest
import tabulate
from .runner import benchmark_v2
from itertools import groupby

def argmin(a):
    return min(range(len(a)), key=lambda x : a[x])

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # code for 1 group
    # passed = terminalreporter.stats.get("passed", [])
    # passed = [x for x in passed if x.user_properties]
    # names = [{"test": x.head_line} for x in passed]
    # user_properties = [ dict(x.user_properties) for x in passed ]
    # table = [ name | user_property for name, user_property in zip(names, user_properties) ]
    # print(tabulate.tabulate(table, headers="keys"))

    # group by x.user_properties.get("group", "")
    passed = terminalreporter.stats.get("passed", [])
    passed = [x for x in passed if x.user_properties]
    for x in passed:
        x.user_properties = dict(x.user_properties)
    passed = sorted(passed, key=lambda x: x.user_properties.get("group", ""))
    for group_name, group in groupby(passed, key=lambda x: x.user_properties.get("group", "")):
        group = list(group)
        names = [{"test": x.head_line} for x in group]
        user_properties = [ dict(x.user_properties) for x in group ]
        # make sure that all user_properties have the same keys
        assert all(x.keys() == user_properties[0].keys() for x in user_properties)
        # cast user_properties list(dict) -> dict(list)
        metric_optim = user_properties[0].get("metric_optim", dict())
        # remove metric_optim and group from user_properties 
        for x in user_properties:
            x.pop("metric_optim", None)
            x.pop("group", None)
        user_properties_t = {k: [dic[k] for dic in user_properties] for k in user_properties[0]}
        # if metric_optim[k] == "min", bold the lowest value
        # user_properties_t = {k: [str(x) for x in v if x == min(v) ] for k, v in user_properties_t.items()}
        for k, v in user_properties_t.items():
            if k not in metric_optim:
                continue
            fun = min if metric_optim[k] == "min" else max
            best_x = fun(v)
            user_properties_t[k] = [
                str(x) if x != best_x else f"\033[31m{x}\033[0m" for x in v 
            ]


        # cast back to list(dict)
        user_properties = [dict(zip(user_properties_t,t)) for t in zip(*user_properties_t.values())]




        # user_properties = [ x.user_properties for x in group]
        # for x in user_properties:
        #     x.pop("group", None)
        table = [ name | user_property for name, user_property in zip(names, user_properties) ]
        # print group_name as title
        print(f"\n  \033[1m{group_name}\033[0m")
        print(tabulate.tabulate(table, headers="keys", tablefmt="rounded_outline"))