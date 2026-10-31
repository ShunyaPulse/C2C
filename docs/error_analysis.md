# Error Analysis & Fairness Slice Report: Chulha to Cylinder (P1)
**Team**: ShunyaPulse  
**Event**: Hacktoberfest Hack Day Lucknow 2026 | AI Community Lucknow  

---

## 1. Where Does the Model Fail?
Analysis of out-of-fold classification errors reveals two distinct failure modes:

### A. The "Fuel Stacking" False Negative (Predicted Clean, Actual Solid Fuel)
- **Profile**: Middle-income rural households with Pucca housing and an active electricity connection that nonetheless continue burning firewood/dung for slow cooking, animal feed boiling, and winter heating.
- **Root Cause**: The survey questions for cooking fuel were removed from the feature set. In rural clusters with extensive free biomass access, affluent homes frequently practice **fuel stacking** (using LPG for quick morning tea, but traditional chulha for primary family meals).
- **Mitigation**: Our **Intervention Priority Index (IPI)** weights land plot area (`land_area_sqfeet`) to flag homes with large rural landholdings where biomass gathering costs are virtually zero.

### B. The "Asset-Poor LPG Adopter" False Positive (Predicted Solid, Actual Clean)
- **Profile**: Lower-income households living in Kutcha structures with very low monthly electricity bills (< ₹150) who have successfully transitioned to exclusive LPG or induction cooking.
- **Root Cause**: Highly targeted subsidy schemes (such as *Pradhan Mantri Ujjwala Yojana*) occasionally enable exclusive clean cooking in economically distressed households, decoupling cooking fuel from standard housing wealth indicators.

---

## 2. Fairness Slice Report (Geographic & Sector Disaggregation)

To ensure algorithmic fairness and evaluate disparate performance across sub-populations, we audited out-of-fold performance across geographic and settlement slices.

### Slice A: State-by-State Disaggregation (5-Fold Out-of-Fold ROC-AUC)

| State | Sample Count | Out-of-Fold ROC-AUC | Solid Fuel Prevalence | Observations |
| :--- | :---: | :---: | :---: | :--- |
| **Chhattisgarh** | 290 | **0.9754** | High (>75%) | High rural homogeneity; clean separation on asset proxies |
| **Odisha** | 488 | **0.9431** | High (>70%) | Strong signal from housing type and grid access |
| **Gujarat** | 594 | **0.9367** | Moderate | Clear wealth divergence between industrial & tribal belts |
| **Kerala** | 290 | **0.9310** | Low (<25%) | High baseline literacy and clean fuel penetration |
| **West Bengal** | 778 | **0.9208** | High (>65%) | Land plot size is strongly predictive of biomass use |
| **Rajasthan** | 776 | **0.9199** | Moderate | Arid/rural clustering creates sharp separation |
| **Maharashtra** | 1,113 | **0.9129** | Moderate | Large sample size; consistent across Vidarbha & Konkan |
| **Uttar Pradesh** | 1,161 | **0.8960** | High (>60%) | Representative central Indo-Gangetic plain distribution |
| **Madhya Pradesh**| 774 | **0.8959** | High (>62%) | Balanced distribution across rural agro-climatic zones |
| **Bihar** | 1,056 | **0.8826** | Very High (>78%) | High baseline solid fuel density lowers relative variance |
| **Karnataka** | 595 | **0.8710** | Moderate | Southern rural/peri-urban gradient |
| **Assam** | 293 | **0.8343** | High (>72%) | Bamboo/tea garden biomass availability creates overlap |
| **Haryana** | 287 | **0.7891** | Low-Mid | Peri-urban agrarian wealth creates ambiguous stacking |
| **Delhi** | 295 | **0.7800** | Very Low (<10%) | Severe class imbalance; near-universal LPG saturation |

### Slice B: Rural vs. Urban Settlement Disaggregation

| Sector | Survey Code (`q103`) | Households | Out-of-Fold ROC-AUC | Calibration Intercept | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rural** | `1` | 9,842 | **0.8495** | -0.012 | 0.114 |
| **Urban / Peri-Urban** | `2` | 1,681 | **0.8235** | +0.038 | 0.098 |

**Interpretation**:
- **Rural Discrimination**: ROC-AUC is higher in rural sectors (0.8495) where physical access, plot size, and construction material clearly separate chulha users from transitioned cylinder households.
- **Urban Calibration**: Urban sectors show slightly lower discrimination due to low baseline solid-fuel prevalence, but display lower absolute Brier score (0.098) reflecting tight probability calibration.

---

## 3. Recommendations for Field Deployment
1. **ASHA / Anganwadi Triage Priority**: Outreach teams should use the top-ranked visit list (`priority_tier == Tier 1`) which captures **88.9% of true solid-fuel households** while filtering out false positives.
2. **Dynamic Calibration Offsets**: For highly urbanized districts (such as Delhi/Chandigarh), apply a -0.15 prior probability shift to account for structural LPG pipeline availability.
