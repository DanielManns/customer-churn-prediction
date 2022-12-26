import argparse


def parse() -> argparse.Namespace:
    """
    Parses arguments.
    :return: argparse.Namespace - parsed arguments
    """

    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-m", "--mode", type=int, default=0, help="Mode of project (0 = train, 1 = postprocessing, 2 = inference)")
    arg("-e", "--exp_name", type=str, help="Experiment name stored in ./experiments as .yaml file", required=True)

    return parser.parse_args()
