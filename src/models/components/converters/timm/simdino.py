import re

def process_simdino(state_dict):
    return {
        k.replace("_orig_mod.", "").replace("backbone.", ""): v
        for k, v in state_dict["teacher"].items()
    }


def process_simdinov2(state_dict):
    state_dict = state_dict["teacher"]
    state_dict = {
        k.removeprefix("backbone."): v
        for k, v in state_dict.items()
        if k.startswith("backbone")
    }
    # revert chunk weight
    state_dict = {
        re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v for k, v in state_dict.items()
    }
    return state_dict