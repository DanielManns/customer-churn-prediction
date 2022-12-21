import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np
import config as c
from src.utility.utility import load_dataset

con = c.config()


def get_preprocessed_dataset(data_type: str, is_subset: bool, input_df: pd.DataFrame = None) -> [pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Returns preprocessed categorical-, continuous-, and mixed DataFrame as well as labels.

    :param data_type: str - data_type for DataFrame filtering
    :param is_subset: bool - subset important variables if True
    :param input_df: pd.DataFrame - Optional input in case of inference, load train data if None
    :return: [pd.DataFrame, pd.DataFrame, ColumnTransformer] - X, y, ColumnTransformer
    """

    if not input_df:
        raw_df = load_dataset()
    else:
        raw_df = input_df
    mixed_df = apply_preprocessing(raw_df)

    y = mixed_df[con.m_config.target_name]
    mixed_df = mixed_df.drop(columns=[con.m_config.target_name])

    # subset important variables
    if is_subset:
        mixed_df = mixed_df.loc[:, con.m_config.im_vars]

    cat_df = mixed_df.drop(columns=get_con_features(mixed_df))
    con_df = mixed_df.drop(columns=get_cat_features(mixed_df))

    if data_type == "categorical":
        X = cat_df
    elif data_type == "continuous":
        X = con_df
    else:
        X = mixed_df

    return X, y


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data cleaning as well as feature engineering to given df.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - preprocessed DataFrame
    """
    return enrich_df(clean_df(df))


def transform_df(X: pd.DataFrame, y: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Transforms Dataframe and returns ColumnTransformer.
    :param X: pd.DataFrame - data
    :param y: pd.DataFrame - labels
    :return: [pd.DataFrame, pd.DataFrame, ColumnTransformer]
    """
    col_transformer = create_col_transformer(X)
    X = col_transformer.fit_transform(X)
    return X, y, col_transformer


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - clean DataFrame
    """

    bool_cols = ["international_plan", "voice_mail_plan", "churn"]
    for col in bool_cols:
        df[col] = df[col].replace("yes", True)
        df[col] = df[col].replace("no", False)

    df["area_code"] = df["area_code"].apply(lambda x: x.replace("area_code_", ""))
    df = df.astype(
        {"state": "category", "area_code": "category", "international_plan": "category", "voice_mail_plan": "category",
         "churn": "category"})
    return df


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches raw DataFrame with additional features used by models.

    :param df: Pandas DataFrame - raw DataFrame
    :return: Pandas DataFrame - raw DataFrame with additional features
    """

    # sum regular calls, minutes and charge
    df["total_reg_calls"] = df["total_eve_calls"] + df["total_night_calls"] + df["total_day_calls"]
    df["total_reg_minutes"] = df["total_eve_minutes"] + df["total_night_minutes"] + df["total_day_minutes"]
    df["total_reg_charge"] = df["total_eve_charge"] + df["total_night_charge"] + df["total_day_charge"]

    # calculate average call duration for each daytime
    df["avg_day_call_duration"] = df["total_day_minutes"].divide(df["total_day_calls"]).round(2)
    df["avg_eve_call_duration"] = df["total_eve_minutes"].divide(df["total_eve_calls"]).round(2)
    df["avg_night_call_duration"] = df["total_night_minutes"].divide(df["total_night_calls"]).round(2)
    df["avg_intl_call_duration"] = df["total_intl_minutes"].divide(df["total_intl_calls"]).round(2)

    avg_group = ["avg_day_call_duration", "avg_eve_call_duration", "avg_night_call_duration", "avg_intl_call_duration"]

    # Fill all na values from zero division
    df[avg_group] = df[avg_group].fillna(value=0.0)

    return df


def create_col_transformer(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create column transformer for sklearn models/pipeline.

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
