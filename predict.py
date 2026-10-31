import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

def clean_survey_codes(df, cols_with_codes=None):
    """
    Standard survey non-responses (88, 99, 999, 9999) represent
    'don't know / refused / not applicable'. Replace them with NaN.
    Guarantees valid numerical columns (e.g. avg_monthly_bill) are never corrupted.
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

def predict_pipeline(input_csv, output_csv, include_reasons=False):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
        
    df = pd.read_csv(input_csv)
    if "household_id" not in df.columns:
        raise ValueError("Input CSV must contain 'household_id' column.")
        
    hh_ids = df["household_id"]
    
    # Load metadata
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(base_dir, "models", "real_metadata.json")
    
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    # Clean codes using exact columns identified in survey dictionary
    cols_with_codes = meta.get("cols_with_codes", None)
    df = clean_survey_codes(df, cols_with_codes=cols_with_codes)
    
    # Load ensemble models
    xgb_path = os.path.join(base_dir, "models", "champion_xgb.joblib")
    lgb_path = os.path.join(base_dir, "models", "champion_lgb.joblib")
    fallback_path = os.path.join(base_dir, "models", "champion_model.joblib")
    
    xgb_model = None
    lgb_model = None
    
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
    elif os.path.exists(fallback_path):
        xgb_model = joblib.load(fallback_path)
        
    if os.path.exists(lgb_path):
        try:
            lgb_model = joblib.load(lgb_path)
        except Exception:
            lgb_model = None
            
    if xgb_model is None and lgb_model is None:
        raise FileNotFoundError(f"Champion model weights missing at {base_dir}/models/.")
        
    feature_names = meta.get("feature_names", [])
    cat_mappings = meta.get("cat_mappings", {})
    
    # If metadata has no feature names, fallback to XGBoost features
    if not feature_names and xgb_model is not None:
        feature_names = getattr(xgb_model, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = feature_names.tolist()
            
    # Construct feature matrix aligned to training schema
    data_dict = {}
    for col in feature_names:
        if col in df.columns:
            if col in cat_mappings:
                cats = cat_mappings[col]
                codes = pd.Categorical(df[col], categories=cats).codes.astype("float64")
                codes[codes == -1] = np.nan  # route unseen categories through missing-value branch
                data_dict[col] = codes
            else:
                data_dict[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            data_dict[col] = np.nan
            
    X = pd.DataFrame(data_dict, index=df.index)
    
    # Predict probabilities (Class 1 = Solid Fuel) using champion blend
    if xgb_model is not None and lgb_model is not None:
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        probs = 0.55 * p_xgb + 0.45 * p_lgb
    elif xgb_model is not None:
        probs = xgb_model.predict_proba(X)[:, 1]
    else:
        probs = lgb_model.predict_proba(X)[:, 1]
        
    # Ensure probabilities are bounded in [1e-4, 1-1e-4] to prevent Log Loss spikes,
    # and kept at 6 decimals to prevent thousands of probability ties in Precision@20% ranking
    probs = np.clip(np.nan_to_num(probs, nan=0.5211), 1e-4, 1.0 - 1e-4)
    probs = np.round(probs, 6)
    
    out_df = pd.DataFrame({
        "household_id": hh_ids,
        "prob_solid_fuel": probs
    })
    
    if include_reasons:
        # Context-aware reasons aligned to predicted probability
        reasons_list = []
        for idx, row in df.iterrows():
            prob = probs[idx]
            r = []
            if prob >= 0.5:
                # Solid fuel drivers
                if row.get("q216_house_pucca_kachha") == 2:
                    r.append("Kachha housing construction")
                if pd.notna(row.get("avg_monthly_bill")) and row.get("avg_monthly_bill") < 250:
                    r.append("Low electricity expenditure (< ₹250)")
                if row.get("q466_fridge_yn") == 0:
                    r.append("No modern refrigeration asset")
                if pd.notna(row.get("land_area_sqfeet")) and row.get("land_area_sqfeet") > 2000:
                    r.append("Large rural plot biomass availability")
                if not r:
                    r.append("Macro demographic solid fuel baseline")
            else:
                # Clean fuel drivers
                if row.get("q216_house_pucca_kachha") == 1:
                    r.append("Pucca modern housing construction")
                if pd.notna(row.get("avg_monthly_bill")) and row.get("avg_monthly_bill") >= 500:
                    r.append("High electricity consumption (>= ₹500)")
                if row.get("q466_fridge_yn") == 1:
                    r.append("Modern refrigeration appliance owned")
                if pd.notna(row.get("land_area_sqfeet")) and row.get("land_area_sqfeet") <= 1000:
                    r.append("Compact residential plot with low biomass")
                if not r:
                    r.append("High socio-economic clean cooking baseline")
            reasons_list.append("; ".join(r[:3]))
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
