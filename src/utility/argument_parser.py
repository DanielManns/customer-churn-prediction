import argparse


def parse() -> argparse.Namespace:
    """
    Parses arguments.
    :return: argparse.Namespace - parsed arguments
    """

    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-r", "--repetitions", type=int, default=2, help="Number of repetitions for each experiment")
    arg("-c", "--classifiers", nargs="+", help="<Required> A minimum of 1 classifiers (sklearn class names)",
        required=False)
    arg("-p", "--classifier_params", nargs="+", help="<Required> A minimum of 1 classifier parameters", required=False)
    arg("-v", "--validation", type=str, default="cross_validate", help="Cross validation method (sklearn class name)")
    # arg("-s", "--is_subset", action="store_true", help="Only use important features if set")
    arg("-e", "--exp_name", type=str, default="test_experiment.yaml", help="Path to experiment configs")

    return parser.parse_args()
