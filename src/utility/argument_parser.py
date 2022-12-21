import argparse


def parse() -> argparse.Namespace:
    """
    Parses arguments.
    :return: argparse.Namespace - parsed arguments
    """

    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-m", "--mode", type=int, default=0, help="Mode of project (0 = train, 1 = postprocessing, 2 = inference)")
    arg("-e", "--exp_names", nargs="+", help="List of experiment names stored in ./experiments as .yaml files", required=True)
    arg("-r", "--repetitions", type=int, default=2, help="Number of repetitions for each experiment")

    return parser.parse_args()
