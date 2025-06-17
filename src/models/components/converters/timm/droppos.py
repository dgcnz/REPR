def preprocess(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}