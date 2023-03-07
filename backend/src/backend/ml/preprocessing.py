import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np
from backend.config import conf, Features, ImpFeatures
from backend.ml.utility import load_train_dataset, load_test_dataset
from typing import Optional
#test 

def get_train_dataset(exp_config: dict) -> list[pd.DataFrame, pd.DataFrame]:
    """
    Returns enriched, but not scaled dataset for given configuration

    :rtype: object
    :param exp_config: dict - experimental configuration
    :return: [pd.DataFrame, pd.DataFrame] - X, y
    """

    X, y = get_clean_dataset(exp_config)
    X = enrich_df(X)

    return X, y

def get_clean_dataset(exp_config: dict) -> list[pd.DataFrame, pd.DataFrame]:
    """
    Returns clean, but not enriched nor scaled dataset for given configuration

    :rtype: object
    :param exp_config: dict - experimental configuration
    :return: [pd.DataFrame, pd.DataFrame] - X, y
    """

    raw_df = load_train_dataset()
    X = clean_df(raw_df)

    y = X[conf.target_name]
    X = X.drop(columns=[conf.target_name])

    features = get_exp_features(exp_config)
    X = X.loc[:, features]

    return X, y


def get_example_dataset(exp_config: dict) -> list[pd.DataFrame, pd.DataFrame]:
    X, y = get_clean_dataset(exp_config)

    # mock id
    hash_values = X.apply(lambda x: int(abs(hash(tuple(x))) / 1e14), axis=1)
    X["id"] = hash_values

    # put id in front
    X = X.loc[:, ["id"] + list(X.columns)]

    return X, y


def scale_df(X: pd.DataFrame, col_transformer: ColumnTransformer=None) -> list[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Transforms Dataframe and returns ColumnTransformer.

    :param X: pd.DataFrame - data
    :param y: pd.DataFrame - labels
    :return: [pd.DataFrame, pd.DataFrame, ColumnTransformer]
    """
    if not col_transformer:
        col_transformer = create_scaler(X)
        X = col_transformer.fit_transform(X)
    else:
        X = col_transformer.transform(X)
    return X, col_transformer


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - clean DataFrame
    """

    if "churn" in df.columns:
        bool_cols = ["international_plan", "voice_mail_plan", "churn"]
    else:
        bool_cols = ["international_plan", "voice_mail_plan"]
    for col in bool_cols:
        df[col] = df[col].replace("yes", True)
        df[col] = df[col].replace("no", False)

    df["area_code"] = df["area_code"].apply(lambda x: x.replace("area_code_", ""))
    df = df.astype(
        {"state": "category", "area_code": "category", "international_plan": "category", "voice_mail_plan": "category"})
    if "churn" in df.columns:
        df = df.astype({"churn": "category"})
    return df


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches cleaned DataFrame with additional features used by models.

    :param df: Pandas DataFrame - cleaned DataFrame
    :return: Pandas DataFrame - cleaned and enriched DataFrame
    """
    
    sum_cols = {
        "total_reg_calls": ["total_eve_calls", "total_night_calls", "total_day_calls"],
        "total_reg_minutes": ["total_eve_minutes", "total_night_minutes", "total_day_minutes"],
        "total_reg_charge": ["total_eve_charge","total_night_charge","total_day_charge"]
    }

    div_cols = {
        "avg_day_call_duration": ["total_day_minutes", "total_day_calls"],
        "avg_eve_call_duration": ["total_eve_minutes", "total_eve_calls"],
        "avg_night_call_duration": ["total_night_minutes", "total_night_calls"],
        "avg_intl_call_duration": ["total_intl_minutes", "total_intl_calls"]
    }

    new_cols = {**sum_cols, **div_cols}

    for col_name, req in new_cols.items():
        if set(req).issubset(df.columns):
            if col_name in sum_cols.keys():
                df[col_name] = df.loc[:, req].sum(axis=1)
            elif col_name in div_cols.keys():
                df[col_name] = df.loc[:, req[0]].divide(df.loc[:, req[1]]).round(2)
                df[col_name] = df[col_name].fillna(0.0)

    

    return df


def create_scaler(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create column transformer for sklearn ml/pipeline.

    :param df: Pandas DataFrame - cleaned and enriched DataFrame
    :return: sklearn.compose ColumnTransformer - ColumnTransformer
    """

    numerical_selector = selector(dtype_exclude="category")
    categorical_selector = selector(dtype_include="category")

    num_columns = numerical_selector(df)
    cat_columns = categorical_selector(df)

    cat_trans = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)
    num_trans = StandardScaler()

    col_transformer = ColumnTransformer([
        ("cat_trans", cat_trans, cat_columns),
        ("num_trans", num_trans, num_columns)
    ], verbose_feature_names_out=False)

    return col_transformer


def get_exp_features(exp_config: dict) -> list[str]:
    """
    Returns feature names for given configuration.

    :param exp_config: dict - eperimental configuration
    :return: list[str] - list of feature names
    """

    if exp_config["features"]["is_subset"]:
        features = list(ImpFeatures.__annotations__.keys())
    else:
        features = list(Features.__annotations__.keys())
    return features
