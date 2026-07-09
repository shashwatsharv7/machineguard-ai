def add_engineered_features(df):
    df = df.copy()

    df["temperature_difference"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["power_proxy"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    )
    return df