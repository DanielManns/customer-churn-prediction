from src.models.postprocessing import run_postprocessing, run_postprocessing_session
from src.utility.argument_parser import parse
import warnings
from src.config import config
from src.models.experiment import run_experiment_session


def warn(*args, **kwargs):
    pass


def start_training(exp_names: [str], reps: int) -> None:
    """
    Starts training of given experiment names.

    :param exp_names: [str] - experiment names
    :param reps: int - number of repetitions for each experiment
    :return: None
    """
    run_experiment_session(exp_names, reps)
    # test


def start_postprocessing(exp_names: [str], reps: int) -> None:
    """
    Starts postprocessing for given experiment names.

    :param exp_names: [str] - experiment names
    :param reps: int - number of repetitions for each experiment
    :return: None
    """
    run_postprocessing_session(exp_names, reps)


def start_inference():
    pass


# TODO: Implement Logger
# TODO: Implement proper results file as pd.DataFrame


if __name__ == "__main__":
    args = parse()
    print(args)
    c = config()

    # supress warnings
    warnings.warn = warn

    if args.mode == 0:
        start_training(args.exp_names, args.repetitions)
    elif args.mode == 1:
        start_postprocessing(args.exp_names, args.repetitions)
    elif args.mode == 2:
        start_inference()
    else:
        raise ValueError("Unexpected project mode! Expected one of [0,1,2]")

