import sho

experiments = {"sho": sho}


def lookup_params(experiment: str, params: list):
    resolved_params = {}
    experiment = experiments[experiment]
    for param in params:
        p_name, p_id = param.split("=")
        choices = experiment.__getattribute__(p_name)
        resolved_params[p_name] = choices[p_id]
    return resolved_params
