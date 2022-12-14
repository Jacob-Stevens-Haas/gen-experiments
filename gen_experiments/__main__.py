import argparse
from pathlib import Path

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
    "--debug",
    action="store_true",
    help=(
        "Run in debug mode, allowing one to use uncommitted code changes and not"
        " recording results"
    ),
)
parser.add_argument("--seed", type=int, help="Random seed for the trial", required=True)
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
ex, group = gen_experiments.experiments[args.experiment]
seed = args.seed
params = gen_experiments.lookup_params(args.experiment, args.param)
trials_folder = Path(__file__).parent.absolute() / "trials"
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
)
