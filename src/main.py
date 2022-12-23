from src.backend.utility.argument_parser import parse
import warnings
from src.config import config
from src.backend.ml.experiment import run_training, predict
from src.backend.utility.utility import create_exp_dirs, load_exp_config
from src.frontend.dashboard import run_gui


def warn(*args, **kwargs):
    pass


# TODO: Implement Logger
# TODO: Implement proper results file as pd.DataFrame


if __name__ == "__main__":
    args = parse()
    print(args)

    # init
    c = config()
    create_exp_dirs(args.exp_name)
    exp_config = load_exp_config(args.exp_name)

    # supress warnings
    warnings.warn = warn

    if args.mode == 0:
        run_training(exp_config)
    elif args.mode == 1:
        predict(exp_config)
    elif args.mode == 2:
        run_gui(exp_config)
    else:
        raise ValueError("Unexpected project mode! Expected one of [0,1,2]")

