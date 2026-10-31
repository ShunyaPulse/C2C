# Technical Decisions Log: Chulha to Cylinder (P1)
**Team**: ShunyaPulse  
**Event**: Hacktoberfest Hack Day Lucknow 2026 | AI Community Lucknow  

---

## 1. Problem Framing & Loss Function Asymmetry
- **Public Health Cost Asymmetry**: In household clean-cooking interventions, errors are not symmetric. A **false negative** (predicting a home uses clean fuel when they actually burn toxic biomass) leaves mothers and children exposed to harmful indoor particulate matter (PM2.5) without government refill support. Conversely, a **false positive** merely expends field outreach effort.
- **Decision**: Calibrate the model to prioritize high **Solid Fuel Recall (88.9%)** and optimize **ROC-AUC (0.9201)** rather than chasing deceptive raw accuracy.

---

## 2. Anticipating Domain Shift (Unseen Evaluation Districts)
- **Challenge**: The participant guide explicitly warned that the hidden test set contains households from **districts not present in the training set**.
- **Risk**: Deep gradient boosted trees easily memorize geographic IDs (`s_name`, `state_abbv`), yielding inflated local validation scores that catastrophically collapse on unseen districts.
- **Decision**:
  1. Constrained tree depth (`max_depth: 5`, `num_leaves: 31`).
  2. Subsampled features (`colsample_bytree: 0.85`) and rows (`subsample: 0.85`).
  3. Added explicit L1 and L2 weight penalties (`reg_alpha=0.5, reg_lambda=2.0`) to regularize leaf weights against regional noise.
  4. Dropped redundant duplicate `state_abbv` (100% collinear duplicate of `s_name`).
  5. Ensured primary splitting signal originates from structural socio-economic indicators (land plot size, housing type, electricity connection years, asset ownership) that generalize uniformly across Indian geography.

---

## 3. Deliberate Treatment of Survey Codes (88 / 99 / 999 / 9999)
- **Context**: 39 columns in `data_dictionary.csv` contain survey non-response codes (`88` = refused, `99` = don't know, `999`/`9999` = not applicable).
- **Risk**: If ingested as raw numeric values, linear models and tree splits treat `9999` as an astronomically large expenditure or age, distorting linear boundaries and creating spurious splits. Conversely, blindly replacing `88`/`99` in continuous columns like `avg_monthly_bill` corrupts valid ₹88 electricity bills into NaNs.
- **Decision**: Extracted the exact 39 columns with `has_code_88_or_99 == True` from `data_dictionary.csv`, cleaned all 88/99 codes into `np.nan` prior to preprocessing, and exported `cols_with_codes` into `real_metadata.json` so `predict.py` and `predict_short.py` apply the exact same clean mask without corrupting valid expenditure amounts. Gradient boosting algorithms natively allocate missing values to the optimal branch direction during training without introducing false numeric artifacts.

---

## 4. Model Architecture & Benchmarking
- **Baseline**: L2-regularized Logistic Regression ($C=0.1$) with median imputation and z-score standard scaling, fit strictly on training folds inside the CV loop to prevent data leakage. Achieved **0.8868 ROC-AUC**.
- **Tree Ensembles**: Evaluated histogram-based XGBoost (**0.9194 ROC-AUC**) and LightGBM (**0.9189 ROC-AUC**).
- **Champion Ensemble Blend**: A weighted combination ($0.55 \times \text{XGBoost} + 0.45 \times \text{LightGBM}$) achieved **0.9201 5-fold CV ROC-AUC**, **0.3593 Log Loss**, and **84.8% accuracy**, significantly outperforming the linear baseline while smoothing individual tree variance. Both final models are exported and blended at inference time.

---

## 5. Honest Negative Results (What Didn't Work)
1. **Deep Trees (`max_depth >= 8`)**: Initial experiments with deeper trees reached 0.94+ training AUC but dropped to 0.87 on cross-validation due to memorization of sparse appliance usage columns (>90% missing).
2. **Mean / Mode Imputation on Survey Codes**: Imputing mean values into non-response codes flattened variance and reduced recall by 2.4 percentage points compared to native missingness routing in tree estimators.
3. **Target Encoding on State Names**: While yielding a marginal bump on in-fold data, target encoding over-relies on state identity and fails under out-of-distribution regional shifts.

