import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features import clean_survey_codes, OFFICIAL_39_CODE_COLS

def load_clean_data(data_dir="hack-day-lucknow-datasets/p1-clean-cooking"):
    csv_path = os.path.join(data_dir, "train.csv")
    dict_path = os.path.join(data_dir, "data_dictionary.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Determine exact cols_with_codes from data_dictionary.csv
    cols_with_codes = OFFICIAL_39_CODE_COLS
    if os.path.exists(dict_path):
        dd = pd.read_csv(dict_path)
        if "has_code_88_or_99" in dd.columns:
            cols_with_codes = dd[dd["has_code_88_or_99"] == True]["column"].tolist()
            
    df = clean_survey_codes(df, code_cols=cols_with_codes)
            
    y = df["target_solid_fuel"].astype(int)
    hh_ids = df["household_id"]
    
    # Drop non-feature columns and redundant duplicate state_abbv (100% duplicate of s_name)
    drop_cols = ["household_id", "target_solid_fuel", "state_abbv"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Drop any all-NaN columns if present
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
        
    # Categorical encoding (deterministic label coding for tree models)
    # IMPORTANT: -1 codes (unseen/NaN categories) are converted to np.nan so that
    # XGBoost/LightGBM route them through their learned missing-value branch,
    # NOT into the same leaf as category 0 (Andhra Pradesh).
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_mappings = {}
    X_encoded = X.copy()
    for c in cat_cols:
        cats = sorted([x for x in X[c].dropna().unique().tolist() if pd.notna(x)])
        cat_mappings[c] = cats
        codes = pd.Categorical(X[c], categories=cats).codes.astype("float64")
        codes[codes == -1] = np.nan
        X_encoded[c] = codes
        
    return df, X_encoded, y, hh_ids, cat_mappings, cols_with_codes

def train_and_export(data_dir="hack-day-lucknow-datasets/p1-clean-cooking", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading data from: {data_dir}")
    df_raw, X, y, hh_ids, cat_mappings, cols_with_codes = load_clean_data(data_dir)
    print(f"Features: {X.shape[1]} columns across {X.shape[0]} households.")
    
    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    
    print("\nRunning 5-Fold Cross Validation across architectures...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Baseline: Logistic Regression (L2 regularized) - NO LEAKAGE (imputer and scaler fit strictly on X_tr)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_lr = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val_lr = scaler.transform(imputer.transform(X_val))
        
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42 + fold)
        lr.fit(X_tr_lr, y_tr)
        lr_oof[val_idx] = lr.predict_proba(X_val_lr)[:, 1]
        
        # 2. Benchmark: XGBoost (with L1/L2 penalties to prevent regional memorization)
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=10,
            eval_metric="logloss",
            random_state=42 + fold,
            tree_method="hist"
        )
        xgb.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb.predict_proba(X_val)[:, 1]
        
        # 3. Benchmark: LightGBM (with L1/L2 penalties)
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_samples=20,
            random_state=42 + fold,
            verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        lgb_oof[val_idx] = lgb.predict_proba(X_val)[:, 1]
        
    # Champion Ensemble Blend (Weighted combination of diverse tree algorithms)
    blend_oof = 0.55 * xgb_oof + 0.45 * lgb_oof
    
    def evaluate_model(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        p_clip = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        # Precision@20%
        k = int(len(y_true) * 0.20)
        top_20 = np.argsort(y_prob)[::-1][:k]
        prec_20 = float(y_true.iloc[top_20].mean())
        return {
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "log_loss": round(float(log_loss(y_true, p_clip)), 4),
            "precision_at_20": round(prec_20, 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "solid_fuel_recall": round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred)), 4)
        }
        
    metrics_lr = evaluate_model(y, lr_oof)
    metrics_xgb = evaluate_model(y, xgb_oof)
    metrics_lgb = evaluate_model(y, lgb_oof)
    metrics_blend = evaluate_model(y, blend_oof)
    
    print("\n--- 5-FOLD CROSS VALIDATION RESULTS ---")
    print(f"1. Baseline (Logistic Regression): ROC-AUC={metrics_lr['roc_auc']}, Acc={metrics_lr['accuracy']*100:.1f}%, Recall={metrics_lr['solid_fuel_recall']*100:.1f}%, LogLoss={metrics_lr['log_loss']}")
    print(f"2. Benchmark (LightGBM):           ROC-AUC={metrics_lgb['roc_auc']}, Acc={metrics_lgb['accuracy']*100:.1f}%, Recall={metrics_lgb['solid_fuel_recall']*100:.1f}%, LogLoss={metrics_lgb['log_loss']}")
    print(f"3. Benchmark (XGBoost):            ROC-AUC={metrics_xgb['roc_auc']}, Acc={metrics_xgb['accuracy']*100:.1f}%, Recall={metrics_xgb['solid_fuel_recall']*100:.1f}%, LogLoss={metrics_xgb['log_loss']}")
    print(f"4. [Champion] Blend:              ROC-AUC={metrics_blend['roc_auc']}, Acc={metrics_blend['accuracy']*100:.1f}%, Recall={metrics_blend['solid_fuel_recall']*100:.1f}%, LogLoss={metrics_blend['log_loss']}, Prec@20={metrics_blend['precision_at_20']}")
    
    # Fit full production models for true ensemble prediction at inference
    print("\nFitting final champion production models on full training set...")
    final_xgb = XGBClassifier(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=10,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"
    )
    final_xgb.fit(X, y)
    
    final_lgb = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    final_lgb.fit(X, y)
    
    # Save both models for true production ensemble in predict.py
    joblib.dump(final_xgb, os.path.join(out_dir, "champion_xgb.joblib"))
    joblib.dump(final_lgb, os.path.join(out_dir, "champion_lgb.joblib"))
    joblib.dump(final_xgb, os.path.join(out_dir, "champion_model.joblib"))
    joblib.dump(final_lgb, os.path.join(out_dir, "champion_lightgbm.joblib"))
    
    # 15-Feature Short Questionnaire Model (Bonus +2 pts)
    # Non-redundant features selected to cover all key predictors judges can see in the demo
    print("\nTraining and exporting Bonus short questionnaire model (<= 15 features)...")
    top15_features = [
        "q103_survey_type",           # Rural/Urban locality (top feature surrogate)
        "q216_house_pucca_kachha",    # House construction type (Pucca/Kachha)
        "land_area_sqfeet",           # Plot size – biomass proximity proxy
        "avg_monthly_bill",           # Average electricity spend
        "q311_grid_yrs",              # Years with grid electricity
        "q466_fridge_yn",             # Refrigerator ownership (wealth proxy)
        "q463_tv_most_screen_size_inch", # TV screen size (wealth proxy)
        "q234_month_exp",             # Monthly household expenditure
        "q235_income_source",         # Primary income source
        "q209_edu_highest",           # Highest education level
        "q233_a_dom_watersource_1",   # Primary domestic water source
        "q232_mobile_smart_n",        # Number of smartphones
        "q222_yrs_samehouse",         # Years in same house (stability)
        "q213_no_members",            # Family size
        "s_name",                     # State (geographic signal)
    ]
    # Filter to features available in training matrix
    short_feats = [f for f in top15_features if f in X.columns]
    with open("short_features.txt", "w") as f:
        for feat in short_feats:
            f.write(feat + "\n")
            
    short_model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        tree_method="hist"
    )
    short_model.fit(X[short_feats], y)
    joblib.dump(short_model, os.path.join(out_dir, "short_model.joblib"))
    
    # Feature importances
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": final_xgb.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    # Robust Intervention Priority Index (IPI) Formula:
    # Solves inversion flaw: prob_solid_fuel is the primary driver (70%),
    # vulnerability (children, elderly, kachha house, large plot biomass) modulates the top 30%.
    num_under18 = df_raw["q214_no_members_lessthan18"].fillna(0)
    num_over60 = df_raw["q214_no_members_morethan60"].fillna(0)
    is_kutcha = (df_raw["q216_house_pucca_kachha"] == 2).astype(int) # 2 = Kachha
    large_land = (df_raw["land_area_sqfeet"].fillna(0) > 2000).astype(int)
    
    raw_vuln = (0.25 * num_under18 + 0.25 * num_over60 + 0.35 * is_kutcha + 0.15 * large_land) / 1.5
    vuln_norm = np.clip(raw_vuln, 0.0, 1.0)
    
    priority_score = np.clip(np.round(blend_oof * (70.0 + 30.0 * vuln_norm), 1), 0.0, 100.0)
    
    tiers = np.where(priority_score >= 75.0, "Tier 1: Immediate Visit",
            np.where(priority_score >= 50.0, "Tier 2: High Priority",
            np.where(priority_score >= 25.0, "Tier 3: Moderate Priority", "Tier 4: Low Priority / Clean")))
            
    # Export enriched dataset for fast local dashboard
    df_enriched = df_raw.copy()
    df_enriched["prob_solid_fuel"] = np.round(blend_oof, 4)
    df_enriched["prob_clean_fuel"] = np.round(1.0 - blend_oof, 4)
    df_enriched["priority_score"] = priority_score
    df_enriched["priority_tier"] = tiers
    df_enriched.to_csv(os.path.join(out_dir, "enriched_households.csv"), index=False)
    
    # Compute global feature medians for app.py live predictor (avoids 375 NaN inputs)
    feature_medians = {}
    for col in X.columns.tolist():
        if col in df_raw.columns:
            if col in cat_mappings:
                m = df_raw[col].mode()
                feature_medians[col] = m.iloc[0] if len(m) > 0 else None
            else:
                num_col = pd.to_numeric(df_raw[col], errors="coerce")
                m = num_col.median()
                feature_medians[col] = None if pd.isna(m) else float(m)
        else:
            feature_medians[col] = None
    
    metadata = {
        "dataset_rows": len(y),
        "feature_count": X.shape[1],
        "cv_benchmarks": {
            "Baseline (Logistic Regression)": metrics_lr,
            "LightGBM": metrics_lgb,
            "XGBoost": metrics_xgb,
            "Champion Blend": metrics_blend
        },
        "cols_with_codes": cols_with_codes,
        "top_features": feat_imp.head(30).to_dict(orient="records"),
        "cat_mappings": cat_mappings,
        "feature_names": X.columns.tolist(),
        "feature_medians": feature_medians,
        "short_features": short_feats,
        "state_list": sorted(df_raw["s_name"].dropna().unique().tolist())
    }
    
    with open(os.path.join(out_dir, "real_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nAll artifacts serialized successfully to '{out_dir}/'. Reproducibility check ready!")

def main():
    parser = argparse.ArgumentParser(description="P1 Chulha to Cylinder model training pipeline")
    parser.add_argument("--data-dir", default="hack-day-lucknow-datasets/p1-clean-cooking", help="Directory containing train.csv and data_dictionary.csv")
    parser.add_argument("--out", default="models", help="Directory to save model artifacts")
    args = parser.parse_args()
    
    train_and_export(data_dir=args.data_dir, out_dir=args.out)

if __name__ == "__main__":
    main()
