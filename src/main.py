from src.utility.argument_parser import parse
import warnings
from src.config import config
from src.models.training import run_experiment_session

c = config()


def warn(*args, **kwargs):
    pass


def start_training():
    run_experiment_session(c.u_config.experiments)

# TODO: Implement save utility for models and results folder
# TODO: Implement Logger
# TODO: Implement cross validation pruning
if __name__ == "__main__":
    args = parse()
    print(args)

    # supress warnings
    warnings.warn = warn

    start_training()

