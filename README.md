# 🔥 Chulha to Cylinder (C2C)
### AI-Powered Household Energy Transition & Field Outreach Prioritization System
**Hacktoberfest Hack Day Lucknow 2026 | AI Community Lucknow**  
*Problem Statement 01: Tabular Machine Learning*  
*Dataset: Official Hack Day Lucknow Real Household Survey (11,523 households across 21 Indian states, 395 survey features)*

---

## 📌 Executive Summary
In India, millions of households continue to cook on solid fuels (*chulha* using firewood, dung cake, crop residue, and coal) despite the nationwide availability of subsidized LPG under *Pradhan Mantri Ujjwala Yojana*. The hazardous smoke produced by biomass cooking causes catastrophic indoor air pollution, disproportionately sickening women and young children.

**Chulha to Cylinder (C2C)** is an end-to-end machine learning system paired with an interactive decision-support application. It predicts household clean fuel adoption probability from survey indicators, demystifies structural drivers through explainable AI, and delivers an actionable **Outreach Triage Engine** to help public health and clean-cooking teams visit high-risk homes first.

---

## 🎯 Problem Statement & Hackathon Alignment

| Requirement | Implementation in C2C | Status |
| :--- | :--- | :---: |
| **1. Classification Model** | Gradient Boosted Trees (XGBoost + LightGBM Blend) trained on 394 survey features | ✅ **0.9201 ROC-AUC / 88.8% Recall** |
| **2. Interactive Prediction Tool** | Real-time simulator with interactive sliders, dynamic gauge chart, and risk thresholds | ✅ **Complete in Streamlit** |
| **3. Factor Explanation** | Global feature attribution showing top structural determinants driving energy transition | ✅ **Complete (Plotly Bar)** |
| **Bonus 1: Single Household Drivers** | Local feature impact waterfall explaining exactly *why* an individual household is at risk | ✅ **Complete (Local Explainability)** |
| **Bonus 2: Field Visit Prioritization** | **Intervention Priority Index (IPI)** ranking homes by solid fuel risk + child/elderly vulnerability | ✅ **Complete with 1-Click CSV Download** |
| **Bonus 3: Analytical Dashboard** | Regional and demographic breakdown across 21 Indian states, housing types, and land area | ✅ **Complete (State & Asset EDA)** |

---

## 📊 Cross-Validation Benchmarks (5-Fold Stratified)

| Model Architecture | 5-Fold ROC-AUC | Accuracy | Solid Fuel Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.9189 | 84.80% | 88.84% | 0.8590 |
| **XGBoost (Histogram)** | 0.9194 | 84.72% | 88.75% | 0.8582 |
| **🏆 Champion Ensemble Blend** | **0.9201** | **84.68%** | **88.78%** | **0.8580** |

> **Why Solid Fuel Recall Matters:**  
> In public health outreach, a **false negative** (predicting a home has clean fuel when they actually burn toxic biomass) leaves young children and mothers breathing toxic indoor PM2.5. Our model architecture is tuned for high recall (>88.8%) to prevent vulnerable homes from falling through the cracks.

---

## 🌐 Top Macro Drivers Identified in the Survey
1. **Land / Plot Area (`land_area_sqfeet`)**: Families with larger plots have direct physical access to tree branches, crop waste, and dung, acting as an economic disincentive for LPG refills.
2. **Housing Construction (`q216_house_pucca_kachha`)**: Kutcha housing is strongly correlated with continued chulha usage.
3. **Appliance Ownership (`q466_fridge_yn`, `q459_tv_yn`, `q463_tv_most_screen_size_inch`)**: Refrigerator and TV screen size act as robust, non-falsifiable proxies for household wealth.
4. **Electricity Reliability & Billing (`q311_grid_yrs`, `avg_monthly_bill`)**: Energy transitions happen across utility clusters; higher electricity grid tenure correlates with clean cooking adoption.

---

## 🚨 Field Outreach Triage Engine (Bonus Idea 2)
Field workers cannot visit 11,523 households individually. C2C computes an **Intervention Priority Index (IPI)**:
$$\text{IPI} = P(\text{Solid Fuel}) \times \left[ 1.0 + 0.30 \cdot (\text{Children } < 18) + 0.25 \cdot (\text{Elderly } 60+) + 0.35 \cdot (\text{Kutcha House}) \right] \times 45$$

- Categorizes households into **Tier 1 (Immediate Visit)**, **Tier 2 (High Priority)**, and **Tier 3**.
- Features an **Instant Filter by State** (e.g. Uttar Pradesh, Bihar, Maharashtra) and a **1-Click Outreach CSV Download** formatted for field workers.

---

## ⚡ Quickstart

### 1. Run the Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view all 4 interactive tabs.

### 2. Generate Predictions / Competition Submissions
```bash
python generate_submission.py [optional_path_to_test.csv]
```
Outputs `submission.csv` in the required format: `household_id,prob_solid_fuel`.
