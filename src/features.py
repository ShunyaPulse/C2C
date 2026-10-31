import numpy as np
import pandas as pd

def clean_survey_codes(df, dict_path=None):
    """
    Cleans non-response codes (88, 99, 999, 9999) across survey columns,
    treating them deliberately as missing values (NaN) to prevent false numeric learning.
    """
    code_cols = [
        "q205_priminc_earner_age", "q211_caste", "q212_ration_card",
        "q222_yrs_samehouse", "q227_a_plot_area", "q228_plot_use_house",
        "q234_month_exp", "q235_income_source", "avg_monthly_bill"
    ]
    df_clean = df.copy()
    for col in code_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
    return df_clean

def prepare_feature_matrix(df, feature_names, cat_mappings):
    """
    Constructs a deterministic feature matrix aligned to the trained model schema.
    """
    data_dict = {}
    for col in feature_names:
        if col in df.columns:
            if col in cat_mappings:
                cats = cat_mappings[col]
                data_dict[col] = pd.Categorical(df[col], categories=cats).codes
            else:
                data_dict[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            data_dict[col] = np.nan
    return pd.DataFrame(data_dict, index=df.index)
