import pytest
import tabulate
from .runner import benchmark_v2
from itertools import groupby


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


def argsort(a):
    return sorted(range(len(a)), key=lambda x: a[x])


class DataFrame(object):
    def __init__(self, rows: list[dict]):
        self.columns = list(rows[0].keys())
        self.data = {k: [r.get(k, None) for r in rows] for k in self.columns}
        
    def remove_columns(self, exclude: list[str]):
        for attr in exclude:
            if attr in self.columns:
                self.columns.remove(attr)
                del self.data[attr] 

    def sort_by(self, column: str, reverse: bool = False):
        idx = argsort(self.data[column])
        if reverse:
            idx = idx[::-1]
        for k in self.data:
            self.data[k] = [self.data[k][i] for i in idx]

    
    def __len__(self):
        return len(self.data[self.columns[0]])
    

    def preprend_column(self, column_name: str, column_data: list):
        self.columns = [column_name] + self.columns
        self.data[column_name] = column_data
        self.data = {k: self.data[k] for k in self.columns}


    
    def print_table(self, column_order: dict[str, str] = {}, highlight_top_k: int = 1):
        """
        Print the table using tabulate.
        Highlight the top k values in each column based on column_order.
        """
        data = self.data.copy() 
        for column_name, order in column_order.items():
            if column_name not in self.columns:
                continue
            idx = argsort(data[column_name])
            if order == "max":
                idx = idx[::-1]

            best_idx = idx[:highlight_top_k]
            for i in range(len(self)):
                if i in best_idx:
                    data[column_name][i] = f"\033[31m{data[column_name][i]}\033[0m"


        table = [{k: data[k][i] for k in self.columns} for i in range(len(self))]
        # print(tabulate.tabulate(table, headers="keys", tablefmt="rounded_outline"))
        print(tabulate.tabulate(table, headers="keys"))





def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = terminalreporter.stats.get("passed", [])
    passed = [x for x in passed if x.user_properties]
    for x in passed:
        x.user_properties = dict(x.user_properties)
    passed = sorted(passed, key=lambda x: x.user_properties.get("group", ""))
    for group_name, group in groupby(
        passed, key=lambda x: x.user_properties.get("group", "")
    ):
        group = list(group)
        test_names = [x.head_line for x in group]
        # user_properties is a list of dictionaries containing both runner options and the run's results
        # [{metric_optim: {metric: min/max, ...}, group: str, ..., "time/mean (ms)"}, ...]
        # user_properties = [dict(x.user_properties) for x in group]

        # make sure that all user_properties have the same keys
        # assert all(x.keys() == user_properties[0].keys() for x in user_properties)

        option_types = {
            "metric_optim": dict, # {metric: min/max, ...}
            "group": str, 
            "drop_columns": list, # [metric, ...]
            "sort_by": str, # metric
        }
        options = {
            k: group[0].user_properties.get(k, default())
            for k, default in option_types.items()
        }

        metric_names = [k for k in options["metric_optim"]]
        runs_metrics = [
            {
                k: v
                for k, v in item.user_properties.items()
                if k in metric_names
            }
            for item in group
        ]

        df = DataFrame(runs_metrics)
        df.remove_columns(options["drop_columns"])
        df.preprend_column("test", test_names)
        if options["sort_by"]:
            df.sort_by(options["sort_by"], reverse=options["metric_optim"][options["sort_by"]] == "max")
        print(f"\n  \033[1m{group_name}\033[0m")
        df.print_table(column_order=options["metric_optim"], highlight_top_k=1)



        

        # runs_metrics_t = {
        #     m: [run_metrics[m] for run_metrics in runs_metrics] for m in metric_names
        # }
        # delete keys from attributes_filter
        # if options["attributes_filter"]:
        #     delete_keys = set(runs_metrics_t.keys()) - set(options["attributes_filter"])
        #     for k in delete_keys:
        #         del runs_metrics_t[k]


        # for k, v in runs_metrics_t.items():
        #     sorted_v = sorted(v, reverse=options["metric_optim"][k] == "max")
        #     best_x = sorted_v[0]
        #     runs_metrics_t[k] = [
        #         str(x) if x != best_x else f"\033[31m{x}\033[0m" for x in v
        #     ]

        # # cast back to list(dict)
        # user_properties = [
        #     dict(zip(runs_metrics_t, t)) for t in zip(*runs_metrics_t.values())
        # ]

        # # sort using sort_by metric
        # # argsort based on

        # table = [
        #     name | user_property for name, user_property in zip(names, user_properties)
        # ]
        # print group_name as title
        # print(tabulate.tabulate(table, headers="keys", tablefmt="rounded_outline"))
