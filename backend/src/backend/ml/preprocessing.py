import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np
from backend.config import conf, Features, ImpFeatures
from backend.ml.utility import load_train_dataset, load_test_dataset
from typing import Optional


def get_train_dataset(exp_config: dict) -> list[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns preprocessed categorical-, continuous-, and mixed DataFrame as well as labels.

    :rtype: object
    :param exp_config: dict - experimental configuration
    :param train: bool - whether to load train dataset
    :return: [pd.DataFrame, pd.DataFrame] - X, y
    """

    y = None

    raw_df = load_train_dataset()
    X = apply_preprocessing(raw_df)
    y = X[conf.target_name]
    X = X.drop(columns=[conf.target_name])

    # subset important variables
    features = get_features(exp_config)
    X = X.loc[:, features]

    return X, y


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data cleaning as well as feature engineering to given df.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - preprocessed DataFrame
    """
    return enrich_df(clean_df(df))


def scale_df(X: pd.DataFrame, y: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Transforms Dataframe and returns ColumnTransformer.

    :param X: pd.DataFrame - data
    :param y: pd.DataFrame - labels
    :return: [pd.DataFrame, pd.DataFrame, ColumnTransformer]
    """
    col_transformer = create_scaler(X)
    X = col_transformer.fit_transform(X)
    return X, y, col_transformer


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
    Enriches raw DataFrame with additional features used by ml.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - raw DataFrame with additional features
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

    :param df: Pandas DataFrame - clean and enriched DataFrame
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


def get_features(exp_config: dict) -> list:
    if exp_config["features"]["is_subset"]:
        features = list(ImpFeatures.__annotations__.keys())
    else:
        features = list(Features.__annotations__.keys())
    return features


def get_cat_features(df: pd.DataFrame) -> list:
    """
    Returns categorical variables of given DataFrame.

    :param df: Pandas DataFrame - DataFrame
    :return: String [] - List of categorical column names
    """

    return list(df.columns[df.dtypes == "category"])


def get_con_features(df: pd.DataFrame) -> list:
    """
    Returns continuous variables of given DataFrame.

    :param df: Pandas DataFrame - DataFrame
    :return: String [] - List of continuous column names
    """
    return list(df.select_dtypes([np.number]).columns)
