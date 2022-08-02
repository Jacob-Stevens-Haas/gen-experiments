import argparse
from pathlib import Path

import mitosis

import exp

parser = argparse.ArgumentParser(
    prog="pysindy-experiments",
    description=(
        "Run defined experiments on simulated data to identify the behavior"
        "of different aspects of the SINDy method."
    ),
)
parser.add_argument("experiment", help="Name to identify the experiment")
parser.add_argument(
    "--debug",
    action="store_true",
    help=(
        "Run in debug mode, allowing one to use uncommitted code changes and not"
        " recording results"
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Random seed for the trial"
    ),
    required=True
)
parser.add_argument(
    "--param",
    action="append",
    help=(
        "Name of parameters to use with this trial, in format 'key=value'\ne.g. --param"
        " seed=1 --param solver=solver_1\nKeys must be understood by the experiment"
        " being run.  Values reference variables\nstored by the same name in the"
        " experiment's namespace"
    ),
)
args = parser.parse_args()
ex = exp.experiments[args.experiment]
seed = args.seed
params = exp.lookup_params(args.experiment, args.param)
trials_folder = Path(__file__).parent.absolute() / "trials"
if not trials_folder.exists():
    trials_folder.mkdir(parents=True)
mitosis.run(
    ex,
    args.debug,
    seed = seed,
    logfile = f"trials_{ex.name}.db",
    params=params,
    trials_folder=trials_folder,
)
