import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches original DataFrame with additional features used by models.

    :param df: Pandas DataFrame - original DataFrame
    :return: Pandas DataFrame - original DataFrame with additional features
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
