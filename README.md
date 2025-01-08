# SINDy Experiments
`pysindy` experiments on Kalman SINDy and single-step SINDy.

# Vendored branch
This branch removes the `pysindy` and `derivative` dependencies
    so that other projects which tie together tight development
            of `pysindy` and `pysindy-experiments`
        can manage these packages' compatible versions.
Do not develop off of this branch!
Instead, fork features off of main and merge into this branch,
    allowing those changes to later be PR'd into main.
