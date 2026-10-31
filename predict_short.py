import os
import argparse
import joblib
import numpy as np
import pandas as pd

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
    
    with open(features_path, "r") as f:
        short_features = [line.strip() for line in f if line.strip()]
        
    model = joblib.load(model_path)
    
    # Load state categories
    state_cats = []
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
            state_cats = meta.get("state_list", [])
            
    # Extract only the 15 short features
    data_dict = {}
    for col in short_features:
        if col in df.columns:
            if col == "s_name":
                data_dict[col] = pd.Categorical(df[col], categories=state_cats).codes
            else:
                data_dict[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            data_dict[col] = np.nan
            
    X = pd.DataFrame(data_dict, index=df.index)
    probs = model.predict_proba(X)[:, 1]
    probs = np.clip(np.nan_to_num(probs, nan=0.521), 0.0, 1.0)
    probs = np.round(probs, 4)
    
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
