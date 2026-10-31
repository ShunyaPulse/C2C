import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

def clean_survey_codes(df, cols_with_codes=None):
    """
    Standard survey non-responses (88, 99, 999, 9999) represent
    'don't know / refused / not applicable'. Replace them with NaN.
    """
    if cols_with_codes is None:
        cols_with_codes = [
            "q205_priminc_earner_age", "q211_caste", "q212_ration_card",
            "q222_yrs_samehouse", "q227_a_plot_area", "q228_plot_use_house",
            "q235_income_source", "q323_kero_mexp", "q475_wash_mach_bee_star"
        ]
    for col in cols_with_codes:
        if col in df.columns:
            df[col] = df[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
    return df

def predict_short(input_csv, output_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
        
    df = pd.read_csv(input_csv)
    if "household_id" not in df.columns:
        raise ValueError("Input CSV must contain 'household_id' column.")
        
    hh_ids = df["household_id"]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, "short_features.txt")
    model_path = os.path.join(base_dir, "models", "short_model.joblib")
    meta_path = os.path.join(base_dir, "models", "real_metadata.json")
    
    if not os.path.exists(features_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Short model or feature list missing in {base_dir}.")
        
    with open(features_path, "r") as f:
        short_features = [line.strip() for line in f if line.strip()]
        
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    cols_with_codes = meta.get("cols_with_codes", None)
    df = clean_survey_codes(df, cols_with_codes=cols_with_codes)
    
    model = joblib.load(model_path)
    
    cat_mappings = meta.get("cat_mappings", {})
    state_cats = cat_mappings.get("s_name", meta.get("state_list", []))
    
    data_dict = {}
    for col in short_features:
        if col in df.columns:
            if col == "s_name":
                codes = pd.Categorical(df[col], categories=state_cats).codes.astype("float64")
                codes[codes == -1] = np.nan  # route unseen states through missing-value branch
                data_dict[col] = codes
            elif col in cat_mappings:
                codes = pd.Categorical(df[col], categories=cat_mappings[col]).codes.astype("float64")
                codes[codes == -1] = np.nan
                data_dict[col] = codes
            else:
                data_dict[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            data_dict[col] = np.nan
            
    X = pd.DataFrame(data_dict, index=df.index)
    probs = model.predict_proba(X)[:, 1]
    probs = np.clip(np.nan_to_num(probs, nan=0.5211), 1e-4, 1.0 - 1e-4)
    probs = np.round(probs, 6)
    
    out_df = pd.DataFrame({
        "household_id": hh_ids,
        "prob_solid_fuel": probs
    })
    
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Short prediction complete using 15 features. Output saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="P1 Short Questionnaire Inference (<=15 features)")
    parser.add_argument("--input", required=True, help="Path to input test CSV")
    parser.add_argument("--output", required=True, help="Path to output predictions CSV")
    args = parser.parse_args()
    
    predict_short(args.input, args.output)

if __name__ == "__main__":
    main()
