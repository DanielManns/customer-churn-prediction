from src.models.postprocessing import run_postprocessing, run_postprocessing_session
from src.utility.argument_parser import parse
import warnings
from src.config import config
from src.models.training import run_experiment_session

c = config()


def warn(*args, **kwargs):
    pass


def start_training():
    run_experiment_session(c.u_config.experiments)


def start_postprocessing():
    run_postprocessing_session(c.u_config.experiments)


def start_inference():
    pass


# TODO: Implement Logger
# TODO: Implement proper results file


if __name__ == "__main__":
    args = parse()
    print(args)

    # supress warnings
    warnings.warn = warn

    start_training()
    # start_postprocessing()
