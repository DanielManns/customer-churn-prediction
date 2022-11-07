import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans original DataFrame.

    :param df: Pandas DataFrame - original DataFrame
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

    cat_trans = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_trans = StandardScaler()

    col_transformer = ColumnTransformer([
        ("cat_trans", cat_trans, cat_columns),
        ("num_trans", num_trans, num_columns)
    ])

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


