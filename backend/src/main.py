import warnings


from backend.api import start_api
from backend.argument_parser import parse
from backend.ml.experiment import train_experiment
from backend.ml.utility import load_exp_config


def warn(*args, **kwargs):
    pass


# TODO: Implement Logger
# TODO: Implement proper results file as pd.DataFrame


if __name__ == "__main__":
    args = parse()

    # init
    exp_config = load_exp_config(args.exp_name)

    # supress warnings
    warnings.warn = warn

    if args.mode == 0:
        train_experiment(exp_config)
        start_api()
    elif args.mode == 1:
        start_api()
        # predict(exp_config)
    # elif args.mode == 2:
    #     run_gui(exp_config)
    else:
        raise ValueError("Unexpected project mode! Expected one of [0,1,2]")

