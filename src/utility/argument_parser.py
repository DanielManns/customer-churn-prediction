import argparse


def parse() -> argparse.Namespace:
    """
    Parses arguments.
    :return: argparse.Namespace - parsed arguments
    """

    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-e", "--exp_names", nargs="+", help="List of experiment name yaml files", required=True)
    arg("-r", "--repetitions", type=int, default=2, help="Number of repetitions for each experiment")
    arg("-v", "--validation", type=str, default="cross_validate", help="Cross validation method (sklearn class name)")
    # arg("-s", "--is_subset", action="store_true", help="Only use important features if set")
    arg("-c", "--classifiers", nargs="+", help="<Required> A minimum of 1 classifiers (sklearn class names)",
        required=False)
    arg("-p", "--classifier_params", nargs="+", help="<Required> A minimum of 1 classifier parameters", required=False)

    return parser.parse_args()
