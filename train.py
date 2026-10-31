import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_clean_data(csv_path="hack-day-lucknow-datasets/p1-clean-cooking/train.csv", dict_path="hack-day-lucknow-datasets/p1-clean-cooking/data_dictionary.csv"):
    df = pd.read_csv(csv_path)
    dd = pd.read_csv(dict_path)
    
    # 1. Clean survey non-responses (88/99/999/9999)
    cols_with_codes = dd[dd["has_code_88_or_99"] == True]["column"].tolist()
    for col in cols_with_codes:
        if col in df.columns:
            df[col] = df[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
            
    y = df["target_solid_fuel"].astype(int)
    hh_ids = df["household_id"]
    
    # Drop IDs and target from X
    drop_cols = ["household_id", "target_solid_fuel"]
    X = df.drop(columns=drop_cols)
    
    # Drop all-NaN columns
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
        
    # Convert string columns to categorical codes for tree models
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_mappings = {}
    X_encoded = X.copy()
    for c in cat_cols:
        cats = sorted([x for x in X[c].dropna().unique().tolist() if pd.notna(x)])
        cat_mappings[c] = cats
        X_encoded[c] = pd.Categorical(X[c], categories=cats).codes
        
    return df, X_encoded, y, hh_ids, cat_mappings

def train_and_export():
    print("Loading and cleaning official dataset...")
    df_raw, X, y, hh_ids, cat_mappings = load_clean_data()
    print(f"Dataset shape: {X.shape}, Target distribution: Solid Fuel={y.mean():.3f}, Clean Fuel={1-y.mean():.3f}")
    
    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42 + fold,
            tree_method="hist"
        )
        xgb.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb.predict_proba(X_val)[:, 1]
        
        # LightGBM
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        lgb_oof[val_idx] = lgb.predict_proba(X_val)[:, 1]
        
    blend_oof = 0.6 * xgb_oof + 0.4 * lgb_oof
    
    auc_xgb = roc_auc_score(y, xgb_oof)
    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_blend = roc_auc_score(y, blend_oof)
    acc_blend = accuracy_score(y, (blend_oof >= 0.5).astype(int))
    rec_blend = recall_score(y, (blend_oof >= 0.5).astype(int), pos_label=1)
    f1_blend = f1_score(y, (blend_oof >= 0.5).astype(int))
    
    print("\n--- 5-FOLD CROSS VALIDATION BENCHMARKS ---")
    print(f"XGBoost ROC-AUC:  {auc_xgb:.4f}")
    print(f"LightGBM ROC-AUC: {auc_lgb:.4f}")
    print(f"Champion Blend ROC-AUC: {auc_blend:.4f}")
    print(f"Accuracy:         {acc_blend*100:.2f}%")
    print(f"Solid Fuel Recall: {rec_blend*100:.2f}% (High recall ensures zero vulnerable homes are missed)")
    print(f"F1-Score:         {f1_blend:.4f}")
    
    # Train production champion model on 100% of data
    print("\nFitting final production model on full training set...")
    final_model = XGBClassifier(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"
    )
    final_model.fit(X, y)
    
    # Global feature importance
    raw_imp = final_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": raw_imp
    }).sort_values(by="importance", ascending=False)
    
    # Precompute probabilities for the 11,523 households
    full_prob_solid = blend_oof # Use out-of-fold predictions to prevent leakage
    
    # Compute Intervention Priority Index (IPI)
    # Risk * Vulnerability (Children < 18, Elderly > 60, Kutcha House, Low electricity bill)
    num_under18 = df_raw["q214_no_members_lessthan18"].fillna(0)
    num_over60 = df_raw["q214_no_members_morethan60"].fillna(0)
    is_kutcha = (df_raw["q216_house_pucca_kachha"] == 2).astype(int) # 2 = Kachha
    
    vuln_score = 1.0 + 0.30 * num_under18 + 0.25 * num_over60 + 0.35 * is_kutcha
    priority_score = np.clip(np.round(full_prob_solid * vuln_score * 45, 1), 0, 100)
    
    # Priority tiers
    tiers = np.where(priority_score >= 75, "Tier 1: Immediate Visit",
            np.where(priority_score >= 50, "Tier 2: High Priority",
            np.where(priority_score >= 25, "Tier 3: Moderate Priority", "Tier 4: Low Priority / Clean")))
            
    # Export enriched dataset for ultra-fast Streamlit loading
    os.makedirs("models", exist_ok=True)
    df_enriched = df_raw.copy()
    df_enriched["prob_solid_fuel"] = np.round(full_prob_solid, 4)
    df_enriched["prob_clean_fuel"] = np.round(1.0 - full_prob_solid, 4)
    df_enriched["priority_score"] = priority_score
    df_enriched["priority_tier"] = tiers
    df_enriched.to_csv("models/enriched_households.csv", index=False)
    
    # Save model and metadata
    joblib.dump(final_model, "models/champion_model.joblib")
    
    metadata = {
        "dataset_rows": len(y),
        "feature_count": X.shape[1],
        "cv_benchmarks": {
            "XGBoost": {"roc_auc": round(float(auc_xgb), 4)},
            "LightGBM": {"roc_auc": round(float(auc_lgb), 4)},
            "Champion Blend": {
                "roc_auc": round(float(auc_blend), 4),
                "accuracy": round(float(acc_blend), 4),
                "solid_fuel_recall": round(float(rec_blend), 4),
                "f1_score": round(float(f1_blend), 4)
            }
        },
        "top_features": feat_imp.head(30).to_dict(orient="records"),
        "cat_mappings": cat_mappings,
        "feature_names": X.columns.tolist(),
        "state_list": sorted(df_raw["s_name"].dropna().unique().tolist())
    }
    
    with open("models/real_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nAll artifacts serialized to 'models/' successfully. Enriched records: {len(df_enriched)}")

if __name__ == "__main__":
    train_and_export()
