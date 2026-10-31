import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

def clean_survey_codes(df):
    """
    Standard survey non-responses (88, 99, 999, 9999) represent
    'don't know / refused / not applicable'. Replace them with NaN.
    """
    code_cols = [
        "q205_priminc_earner_age", "q211_caste", "q212_ration_card",
        "q222_yrs_samehouse", "q227_a_plot_area", "q228_plot_use_house",
        "q234_month_exp", "q235_income_source", "avg_monthly_bill"
    ]
    for col in code_cols:
        if col in df.columns:
            df[col] = df[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
    return df

def predict_pipeline(input_csv, output_csv, include_reasons=False):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
        
    df = pd.read_csv(input_csv)
    if "household_id" not in df.columns:
        raise ValueError("Input CSV must contain 'household_id' column.")
        
    hh_ids = df["household_id"]
    
    # Clean codes
    df = clean_survey_codes(df)
    
    # Load model and metadata
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "champion_model.joblib")
    meta_path = os.path.join(base_dir, "models", "real_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Champion model weights missing at {model_path}.")
        
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    feature_names = meta["feature_names"]
    cat_mappings = meta["cat_mappings"]
    
    # Construct feature matrix aligned to training schema
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
            
    X = pd.DataFrame(data_dict, index=df.index)
    
    # Predict probabilities (Class 1 = Solid Fuel)
    probs = model.predict_proba(X)[:, 1]
    # Ensure probabilities are bounded in [0, 1] with zero NaNs
    probs = np.clip(np.nan_to_num(probs, nan=0.521), 0.0, 1.0)
    probs = np.round(probs, 4)
    
    out_df = pd.DataFrame({
        "household_id": hh_ids,
        "prob_solid_fuel": probs
    })
    
    if include_reasons:
        # Dynamic top reasons generation per household
        reasons_list = []
        for idx, row in df.iterrows():
            r = []
            if row.get("q216_house_pucca_kachha") == 2:
                r.append("Kachha housing")
            if pd.notna(row.get("avg_monthly_bill")) and row.get("avg_monthly_bill") < 250:
                r.append("Low electricity expenditure")
            if row.get("q466_fridge_yn") == 0:
                r.append("No refrigerator asset")
            if pd.notna(row.get("land_area_sqfeet")) and row.get("land_area_sqfeet") > 2000:
                r.append("Large plot biomass availability")
            reasons_list.append("; ".join(r[:3]) if r else "Macro demographic baseline")
        out_df["top_reasons"] = reasons_list
        
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Prediction complete. Output written to {output_csv} ({len(out_df)} rows).")

def main():
    parser = argparse.ArgumentParser(description="P1 Chulha to Cylinder inference entrypoint")
    parser.add_argument("--input", required=True, help="Path to input test CSV")
    parser.add_argument("--output", required=True, help="Path to output predictions CSV")
    parser.add_argument("--reasons", action="store_true", help="Include top_reasons bonus column")
    args = parser.parse_args()
    
    predict_pipeline(args.input, args.output, include_reasons=args.reasons)

if __name__ == "__main__":
    main()
