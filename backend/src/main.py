import warnings


from backend.api import start_api
from backend.argument_parser import parse
from backend.ml.experiment import train_experiment, explain_experiment
from backend.ml.utility import load_exp_config
from backend.ml.preprocessing import get_clean_dataset

EXP_CONFIG = {}

def warn(*args, **kwargs):
    pass


if __name__ == "__main__":
    args = parse()

    # init
    EXP_CONFIG = load_exp_config(args.exp_name)

    # supress warnings
    warnings.warn = warn

    if args.mode == 0:
        train_experiment(EXP_CONFIG)
        start_api()
    elif args.mode == 1:
        start_api()
        # predict(exp_config)
    else:
        raise ValueError("Unexpected project mode! Expected one of [0,1,2]")

