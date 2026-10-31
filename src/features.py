import numpy as np
import pandas as pd

# The 39 exact columns flagged in data_dictionary.csv with has_code_88_or_99 == True
OFFICIAL_39_CODE_COLS = [
    "q205_priminc_earner_age", "q211_caste", "q212_ration_card", "q222_yrs_samehouse",
    "q227_a_plot_area", "q228_plot_use_house", "q236_income_category", "q304_grid_patternpowercut",
    "q314_b_online_pay_easy_diff", "q315_b_bill_jul_oct", "q323_kero_mexp", "q328_comp_6months",
    "q414_ceiling_fan_mostused_brand", "q420_table_fan_mostused_brand", "q426_air_coolers_mostused_brand",
    "q433_ac_most_new_second_hand", "q434_ac_most_bee_rating", "q442_ac_2most_bee_rating",
    "q443_ac_freq_tech_serv", "q444_ac_in2yrs_yn", "q456_geyser_most_bee_star", "q461_tv_most_type",
    "q469_fridge_most_bee_star", "q475_wash_mach_bee_star", "q479_water_pump_bee_star",
    "q607_a_prep_control_elec_exp", "q607_b_prep_elec_bill_change", "q607_c_prep_wrong_bill",
    "q607_d_prep_elec_theft", "q607_e_prep_trust_increase", "q608_a_ifprep_control_elec_exp",
    "q608_b_ifprep_elec_bill_change", "q608_c_ifprep_wrong_bill", "q608_d_ifprep_elec_theft",
    "q608_e_if_trust_increase", "q609_prepaid_meter_int", "q610_b_billed_units",
    "q610_c_billed_amount", "q610_c_arrears"
]

def clean_survey_codes(df, code_cols=None):
    """
    Cleans non-response codes (88, 99, 999, 9999) across survey columns,
    treating them deliberately as missing values (NaN) to prevent false numeric learning.
    Only touches columns where data_dictionary.csv has has_code_88_or_99 == True.
    """
    if code_cols is None:
        code_cols = OFFICIAL_39_CODE_COLS
        
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
