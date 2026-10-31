import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_clean_data(csv_path="hack-day-lucknow-datasets/p1-clean-cooking/train.csv", dict_path="hack-day-lucknow-datasets/p1-clean-cooking/data_dictionary.csv"):
    df = pd.read_csv(csv_path)
    dd = pd.read_csv(dict_path)
    
    # 1. Clean survey non-responses (88/99/999/9999) - treat deliberately as missing
    cols_with_codes = dd[dd["has_code_88_or_99"] == True]["column"].tolist()
    for col in cols_with_codes:
        if col in df.columns:
            df[col] = df[col].replace([88, 99, 999, 9999, 88.0, 99.0, 999.0, 9999.0], np.nan)
            
    y = df["target_solid_fuel"].astype(int)
    hh_ids = df["household_id"]
    
    # Drop non-feature columns
    drop_cols = ["household_id", "target_solid_fuel"]
    X = df.drop(columns=drop_cols)
    
    # Drop all-NaN columns if any
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
        
    # Categorical encoding (deterministic label coding for tree models)
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_mappings = {}
    X_encoded = X.copy()
    for c in cat_cols:
        cats = sorted([x for x in X[c].dropna().unique().tolist() if pd.notna(x)])
        cat_mappings[c] = cats
        X_encoded[c] = pd.Categorical(X[c], categories=cats).codes
        
    return df, X_encoded, y, hh_ids, cat_mappings

def train_and_export():
    print("Loading and cleaning official survey dataset...")
    df_raw, X, y, hh_ids, cat_mappings = load_clean_data()
    print(f"Features: {X.shape[1]} columns across {X.shape[0]} households.")
    
    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    
    # Preprocessing pipeline for Logistic Regression Baseline
    # Impute missing values with median, scale continuous features
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed_scaled = scaler.fit_transform(imputer.fit_transform(X))
    
    print("\nRunning 5-Fold Cross Validation across architectures...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Baseline: Logistic Regression (L2 regularized)
        X_tr_lr, X_val_lr = X_imputed_scaled[train_idx], X_imputed_scaled[val_idx]
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42 + fold)
        lr.fit(X_tr_lr, y_tr)
        lr_oof[val_idx] = lr.predict_proba(X_val_lr)[:, 1]
        
        # 2. Benchmark: XGBoost (Regularized, shallow depth to prevent overfitting)
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
        
        # 3. Benchmark: LightGBM
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
        
    # Champion Ensemble Blend (Weighted combination of diverse tree algorithms)
    blend_oof = 0.55 * xgb_oof + 0.45 * lgb_oof
    
    def evaluate_model(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "solid_fuel_recall": round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred)), 4)
        }
        
    metrics_lr = evaluate_model(y, lr_oof)
    metrics_xgb = evaluate_model(y, xgb_oof)
    metrics_lgb = evaluate_model(y, lgb_oof)
    metrics_blend = evaluate_model(y, blend_oof)
    
    print("\n--- 5-FOLD CROSS VALIDATION RESULTS ---")
    print(f"1. Baseline (Logistic Regression): ROC-AUC={metrics_lr['roc_auc']}, Acc={metrics_lr['accuracy']*100:.1f}%, Recall={metrics_lr['solid_fuel_recall']*100:.1f}%")
    print(f"2. Benchmark (LightGBM):           ROC-AUC={metrics_lgb['roc_auc']}, Acc={metrics_lgb['accuracy']*100:.1f}%, Recall={metrics_lgb['solid_fuel_recall']*100:.1f}%")
    print(f"3. Benchmark (XGBoost):            ROC-AUC={metrics_xgb['roc_auc']}, Acc={metrics_xgb['accuracy']*100:.1f}%, Recall={metrics_xgb['solid_fuel_recall']*100:.1f}%")
    print(f"4. [Champion] Blend:              ROC-AUC={metrics_blend['roc_auc']}, Acc={metrics_blend['accuracy']*100:.1f}%, Recall={metrics_blend['solid_fuel_recall']*100:.1f}%")
    
    # Fit full production model
    print("\nFitting final champion production model on full training set...")
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
    
    # Feature importances
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    # Compute Intervention Priority Index (IPI)
    num_under18 = df_raw["q214_no_members_lessthan18"].fillna(0)
    num_over60 = df_raw["q214_no_members_morethan60"].fillna(0)
    is_kutcha = (df_raw["q216_house_pucca_kachha"] == 2).astype(int) # 2 = Kachha
    
    vuln_multiplier = 1.0 + 0.30 * num_under18 + 0.25 * num_over60 + 0.35 * is_kutcha
    priority_score = np.clip(np.round(blend_oof * vuln_multiplier * 45, 1), 0, 100)
    
    tiers = np.where(priority_score >= 75, "Tier 1: Immediate Visit",
            np.where(priority_score >= 50, "Tier 2: High Priority",
            np.where(priority_score >= 25, "Tier 3: Moderate Priority", "Tier 4: Low Priority / Clean")))
            
    # Export enriched dataset for instantaneous app loading
    os.makedirs("models", exist_ok=True)
    df_enriched = df_raw.copy()
    df_enriched["prob_solid_fuel"] = np.round(blend_oof, 4)
    df_enriched["prob_clean_fuel"] = np.round(1.0 - blend_oof, 4)
    df_enriched["priority_score"] = priority_score
    df_enriched["priority_tier"] = tiers
    df_enriched.to_csv("models/enriched_households.csv", index=False)
    
    # Save artifacts
    joblib.dump(final_model, "models/champion_model.joblib")
    
    metadata = {
        "dataset_rows": len(y),
        "feature_count": X.shape[1],
        "cv_benchmarks": {
            "Baseline (Logistic Regression)": metrics_lr,
            "LightGBM": metrics_lgb,
            "XGBoost": metrics_xgb,
            "Champion Blend": metrics_blend
        },
        "top_features": feat_imp.head(30).to_dict(orient="records"),
        "cat_mappings": cat_mappings,
        "feature_names": X.columns.tolist(),
        "state_list": sorted(df_raw["s_name"].dropna().unique().tolist())
    }
    
    with open("models/real_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nAll artifacts serialized successfully to 'models/'. Ready for presentation!")

if __name__ == "__main__":
    train_and_export()
