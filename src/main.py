import pandas as pd
from src.utility.argument_parser import parse

import warnings
from src.config import config

from src.models.training import run_experiment_session

pd.set_option('display.max_columns', 500)
c = config()


def warn(*args, **kwargs):
    pass


def start_training():
    run_experiment_session(c.u_config.experiments)

# TODO: Implement save utility for models and results folder
# TODO: Implement Logger
# TODO: Implement DT pruning
if __name__ == "__main__":
    args = parse()
    print(args)

    # supress warnings
    warnings.warn = warn

    start_training()

