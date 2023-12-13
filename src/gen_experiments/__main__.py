import argparse
from pathlib import Path
import gen_experiments.utils

import mitosis

import gen_experiments

parser = argparse.ArgumentParser(
    prog="pysindy-experiments",
    description=(
        "Run defined experiments on simulated data to identify the behavior"
        "of different aspects of the SINDy method."
    ),
)
parser.add_argument("experiment", help="Name to identify the experiment")
parser.add_argument(
    "--debug", "-d",
    action="store_true",
    help=(
        "Run in debug mode, allowing one to use uncommitted code changes and not"
        " recording results"
    ),
)
parser.add_argument(
    "--eval-param", "-e", 
    type=str,
    action="append",
    help="Parameters directly passed on command line",
)
parser.add_argument(
    "--param", "-p",
    action="append",
    help=(
        "Name of parameters to use with this trial, in format 'key=value'\ne.g."
        "--param solver=solver_1\nKeys must be understood by the experiment"
        " being run.  Values reference variables\nstored by the same name in the"
        " experiment's namespace"
    ),
)
args = parser.parse_args()
ex, group = gen_experiments.experiments[args.experiment]
params = []
if args.eval_param is None:
    args.eval_param == ()
for ep in args.eval_param:
    arg_name, arg_str = ep.split("=")
    arg_val = eval(arg_str)
    params.append(mitosis.Parameter(str(arg_val), arg_name, arg_val))
    if arg_name == "seed":
        seed = arg_val
lookup_dict = ex.lookup_dict if hasattr(ex, "lookup_dict") else None
if args.param is None:
    args.param = ()
params += gen_experiments.lookup_params(args.param, lookup_dict)
trials_folder = gen_experiments.utils.TRIALS_FOLDER
if not trials_folder.exists():
    trials_folder.mkdir(parents=True)
mitosis.run(
    ex,
    args.debug,
    seed=seed,
    group=group,
    logfile=f"trials_{args.experiment}.db",
    params=params,
    trials_folder=trials_folder,
    untracked_params=("plot_prefs", "skinny_specs"),
    addl_mods_and_names=[(gen_experiments.utils, ["NestedDict"])],
)
