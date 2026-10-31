import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

def generate_submission(test_csv_path="hack-day-lucknow-datasets/p1-clean-cooking/train.csv", output_path="submission.csv"):
    if len(sys.argv) > 1:
        test_csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    print(f"Generating submission from: {test_csv_path}")
    if not os.path.exists(test_csv_path):
        print(f"Error: file {test_csv_path} does not exist.")
        return
        
    df = pd.read_csv(test_csv_path)
    if "household_id" not in df.columns:
        raise ValueError("Input file must contain 'household_id' column.")
        
    hh_ids = df["household_id"]
    
    # Load model and metadata
    model_path = "models/champion_model.joblib"
    meta_path = "models/real_metadata.json"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Champion model not found. Run 'python train.py' first.")
        
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    feature_names = meta["feature_names"]
    cat_mappings = meta["cat_mappings"]
    
    # Clean 88/99/999/9999
    dict_path = "hack-day-lucknow-datasets/p1-clean-cooking/data_dictionary.csv"
    if os.path.exists(dict_path):
        dd = pd.read_csv(dict_path)
        cols_with_codes = dd[dd["has_code_88_or_99"] == True]["column"].tolist()
        for col in cols_with_codes:
            if col in df.columns:
                df[col] = df[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
                
    # Prepare features cleanly without fragmentation
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
            
    # Predict probabilities for solid fuel (class 1)
    probs = model.predict_proba(X)[:, 1]
    
    sub = pd.DataFrame({
        "household_id": hh_ids,
        "prob_solid_fuel": np.round(probs, 4)
    })
    
    sub.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path} with {len(sub)} rows.")
    print("Sample preview:")
    print(sub.head())

if __name__ == "__main__":
    generate_submission()
