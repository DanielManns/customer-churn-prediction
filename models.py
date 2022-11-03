import numpy as np
import pandas as pd
import argparse
from IPython.display import display, HTML


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
import yaml

pd.set_option('display.max_columns', 500)

with open("config.yaml") as p:
    config = yaml.safe_load(p)

train_path = config["paths"]["train_path"]
target_name = config["target_name"]
im_vars = config["important_vars"]
train_seed = config["seeds"]["train"]


def clean_df(data):
    bool_cols = ["international_plan", "voice_mail_plan", "churn"]
    for col in bool_cols:
        data[col] = data[col].replace("yes", True)
        data[col] = data[col].replace("no", False)

    data["area_code"] = data["area_code"].apply(lambda x: x.replace("area_code_", ""))
    data = data.astype(
        {"state": "category", "area_code": "category", "international_plan": "category", "voice_mail_plan": "category",
         "churn": "category"})
    return data


def enrich_df(data):
    # sum calls, minutes and charge
    data["total_calls"] = data["total_eve_calls"] + data["total_night_calls"] + data["total_day_calls"] + data["total_intl_calls"]
    data["total_minutes"] = data["total_eve_minutes"] + data["total_night_minutes"] + data["total_day_minutes"] + data[
        "total_intl_minutes"]
    data["total_charge"] = data["total_eve_charge"] + data["total_night_charge"] + data["total_day_charge"] + data[
        "total_intl_charge"]

    # calculate average call duration for each daytime
    data["avg_day_call_duration"] = data["total_day_minutes"].divide(data["total_day_calls"]).round(2)
    data["avg_eve_call_duration"] = data["total_eve_minutes"].divide(data["total_eve_calls"]).round(2)
    data["avg_night_call_duration"] = data["total_night_minutes"].divide(data["total_night_calls"]).round(2)
    data["avg_intl_call_duration"] = data["total_intl_minutes"].divide(data["total_intl_calls"]).round(2)

    avg_group = ["avg_day_call_duration", "avg_eve_call_duration", "avg_night_call_duration", "avg_intl_call_duration"]

    data[avg_group] = data[avg_group].fillna(value=0.0)

    return data


def create_preprocessor(data):
    numerical_selector = selector(dtype_exclude="category")
    categorical_selector = selector(dtype_include="category")

    num_columns = numerical_selector(data)
    cat_columns = categorical_selector(data)

    cat_trans = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_trans = StandardScaler()

    preprocessor = ColumnTransformer([
        ("cat_trans", cat_trans, cat_columns),
        ("num_trans", num_trans, num_columns)
    ])

    return preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-experiment", type=str, default="./experiments/test_experiment.yaml", help="Path to experiment configs")
    arg("--from-checkpoint", dest="checkpoint", action="store_true",
        help="Continue from checkpoint if set")

    args = parser.parse_args()

    raw = pd.read_csv(train_path)

    data = clean_df(raw)
    data = enrich_df(data)

    target = data[target_name]
    data = data.drop(columns=[target_name])

    # subset important variables
    data = data.loc[:, im_vars]

    with open(args.experiment) as p:
        exp_config = yaml.safe_load(p)

    classifier = eval(exp_config["model"]["class_name"])
    eval_method = eval(exp_config["eval"]["method"])

    preprocessor = create_preprocessor(data)
    classifier = classifier()

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    if eval_method.__name__ == "train_test_split":
        X_train, X_test, y_train, y_test = eval_method(data, target, random_state=train_seed)

        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(f"The {eval_method.__name__} score of {pipe['classifier'].__class__.__name__} is: {acc}")

    else:
        X = preprocessor.fit_transform(data)

        cv_result = cross_validate(pipe, data, target, cv=10)
        scores = cv_result["test_score"]
        print(
            f"The {eval_method.__name__} score of {pipe[1].__class__.__name__} is: "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )
