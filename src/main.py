from src.backend.utility.argument_parser import parse
import warnings
from src.config import config
from src.backend.models.experiment import run_training, run_inference, run_gui
from src.backend.utility import create_exp_dirs, load_exp_config


def warn(*args, **kwargs):
    pass


# TODO: Implement Logger
# TODO: Implement proper results file as pd.DataFrame
# TODO: Save learned transformation from training data with model such that inference works


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
        run_inference(exp_config)
    elif args.mode == 2:
        run_gui(exp_config)
    else:
        raise ValueError("Unexpected project mode! Expected one of [0,1,2]")

