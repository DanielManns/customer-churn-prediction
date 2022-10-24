import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate

train_path = "./data/train.csv"
target_name = "churn"


def clean_df(df):
    bool_cols = ["international_plan", "voice_mail_plan", "churn"]
    for col in bool_cols:
        df[col] = df[col].replace("yes", True)
        df[col] = df[col].replace("no", False)

    df["area_code"] = df["area_code"].apply(lambda x: x.replace("area_code_", ""))
    df = df.astype(
        {"state": "category", "area_code": "category", "international_plan": "category", "voice_mail_plan": "category",
         "churn": "category"})
    return df


def enrich_df(df):
    # sum calls, minutes and charge
    df["total_calls"] = df["total_eve_calls"] + df["total_night_calls"] + df["total_day_calls"] + df["total_intl_calls"]
    df["total_minutes"] = df["total_eve_minutes"] + df["total_night_minutes"] + df["total_day_minutes"] + df[
        "total_intl_minutes"]
    df["total_charge"] = df["total_eve_charge"] + df["total_night_charge"] + df["total_day_charge"] + df[
        "total_intl_charge"]

    # calculate average call duration for each daytime
    df["avg_day_call_duration"] = df["total_day_minutes"].divide(df["total_day_calls"]).round(2)
    df["avg_eve_call_duration"] = df["total_eve_minutes"].divide(df["total_eve_calls"]).round(2)
    df["avg_night_call_duration"] = df["total_night_minutes"].divide(df["total_night_calls"]).round(2)
    df["avg_intl_call_duration"] = df["total_intl_minutes"].divide(df["total_intl_calls"]).round(2)

    avg_group = ["avg_day_call_duration", "avg_eve_call_duration", "avg_night_call_duration", "avg_intl_call_duration"]

    df[avg_group] = df[avg_group].fillna(value=0.0)

    return df


def create_preprocessor(df):
    numerical_selector = selector(dtype_exclude="category")
    categorical_selector = selector(dtype_include="category")

    num_columns = numerical_selector(df)
    cat_columns = categorical_selector(df)

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer([
        ("one_hot_encoder", categorical_preprocessor, cat_columns),
        ("standard_scaler", numerical_preprocessor, num_columns)
    ])

    return preprocessor


if __name__ == "__main__":
    cv = True
    raw = pd.read_csv(train_path)

    data = clean_df(raw)
    data = enrich_df(data)

    target = data[target_name]
    data = data.drop(columns=[target_name])

    model = make_pipeline(create_preprocessor(data), LogisticRegression(max_iter=500))

    if not cv:
        data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=14)
        _ = model.fit(data_train, target_train)
        acc = model.score(data_test, target_test)
        print(acc)

    else:
        cv_result = cross_validate(model, data, target, cv=10)
        scores = cv_result["test_score"]
        print(
            "The mean cross-validation accuracy is: "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )






