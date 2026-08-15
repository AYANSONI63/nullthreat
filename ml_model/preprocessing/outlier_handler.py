import pandas as pd 



def compute_outlier_cap(
        df: pd.DataFrame,
        multiplier: float=1.5
) -> dict:
    
    caps = {}

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3-Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        caps[column] = {
            "lower": lower,
            "upper": upper
        }

    return caps 


def apply_outlier_caps(
        df: pd.DataFrame,
        caps: dict
) -> pd.DataFrame:
    
    df = df.copy()


    for column, limits in caps.items():

        lower = limits["lower"]
        upper = limits["upper"]

        df[column] = df[column].clip(
            lower=lower,
            upper=upper
        )


    return df
