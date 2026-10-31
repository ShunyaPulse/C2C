# 8-Minute Presentation Master Script & Judge Strategy
**Hacktoberfest Hack Day Lucknow 2026 | Problem 01: Chulha to Cylinder (C2C)**  
**Team**: ShunyaPulse (Solo Participant)  
**Strict Time Limit**: 8 Minutes (Stopwatch countdown in room)

---

## Time Allocation & Presentation Map

```
  0:00 ─── Phase 1: Problem Framing & Domain Shift (1.5 min) ─── 1:30
  1:30 ─── Phase 2: Architecture, Evidence & Live Demo (3.5 min) ─ 5:00
  5:00 ─── Phase 3: Q&A, Error Analysis & Integrity (3.0 min) ── 8:00
```

---

## Phase 1: Problem Statement & Constraints (0:00 – 1:30)
> **Rubric D1: 5 Points**  
> *Judge Expectation*: Do NOT merely restate the brief. Frame the human persona, the asymmetric cost of errors, and the domain shift to unseen districts.

### Slide 1 & 2: Framing & Domain Shift
**What to say (in English / Hinglish):**
- *"Good afternoon respected judges. I am representing team ShunyaPulse presenting **Chulha to Cylinder (C2C)**."*
- *"In India, over 50% of rural households still burn solid biomass in traditional chulhas, exposing mothers and infants to toxic indoor particulate pollution. State clean-cooking missions have limited administrative bandwidth and can only visit the top 20% most vulnerable homes."*
- *"**The Real Problem & Asymmetric Error Cost:** Upcoming survey rounds deliberately drop direct cooking questions to reduce interview fatigue. If our model makes a **False Negative**, an impoverished mother suffering from smoke inhalation is skipped and remains on chulha. A **False Positive** merely results in an unnecessary visit to a household already using LPG. Hence, we designed our system prioritizing **Precision@20%**, achieving **0.9583**, meaning 19 out of 20 dispatched visits are guaranteed hits."*
- *"**The Domain Shift Challenge:** The evaluator tests on **unseen districts** absent from the training set. A naive model that memorizes district IDs will overfit and collapse. Therefore, we explicitly dropped district identifiers and grounded our model in universal socio-economic invariants: dwelling structure (pucca vs kachha), continuous electricity expenditure, and durable asset footprints."*

### Slide 3: Non-Response Sanitization
**What to say:**
- *"**Data Engineering Trap:** In survey data, codes like 88, 99, and 9999 denote 'refusal' or 'don't know'. A common beginner mistake is global replacement, which ruins continuous variables like electricity bills of ₹88 or ₹99/month, which are common lifeline tariffs for the poorest families. We engineered a targeted 39-column dictionary sanitizer. Furthermore, we routed categorical missing codes `-1` to `NaN`, enabling gradient boosted trees to route missing branches natively instead of grouping them arbitrarily with index 0."*

---

## Phase 2: Features, Architecture & Live Demo (1:30 – 5:00)
> **Rubric D2: 8 Points**  
> *Judge Expectation*: Show everything working live. Tie every feature to code files. Do not show slides only.

### Slide 4 & 5: Regularized Champion Blend (1:30 – 2:15)
**What to say:**
- *"For our modeling strategy, we benchmarked a 4-feature Logistic Regression baseline (0.801 AUC) up to full gradient boosted trees. Our production champion is a regularized ensemble: **$0.55 \times \text{XGBoost} + 0.45 \times \text{LightGBM}$**."*
- *"To eliminate overfitting on 395 survey features, we enforced strict L1/L2 penalties (`reg_alpha=0.5, reg_lambda=2.0`) in XGBoost and limited LightGBM to 31 leaves with `min_child_samples=30`."*
- *"**Cross-Validation Metrics:** In leak-free 5-fold CV, our champion blend achieves **0.9201 ROC-AUC** (beating the evaluator 0.89 anchor), **0.3593 Log Loss** (beating the 0.42 threshold), and **0.9583 Precision@20%**."*
- *"**Bonus Short Model:** We also trained a 15-question short questionnaire model in `predict_short.py`, which achieves **0.8961 AUC**, preserving 97.4% of full-model performance with only 3.8% of the survey questions."*

### Slide 6: Switch to Live Demo (2:15 – 5:00)
> **ACTION**: Alt-Tab to browser with `localhost:8501` open and `https://c2c-app-685823552970.asia-south1.run.app/docs`.

**Live Demo Walkthrough Steps:**
1. **Show Live Streamlit Dashboard (`localhost:8501`):**
   - *"Here is our live field decision system running on the 11,523 survey households."*
   - Click on **Tab 2: Household Risk Predictor & Explainability**:
     - *"An outreach worker selects a state, say Uttar Pradesh, and enters a household with ₹350 monthly bill, Kachha dwelling, and no refrigerator."*
     - *"Instantly, our champion blend computes a calibrated probability (e.g. 88.4%) and displays the **Intervention Priority Index (IPI)**."*
     - *"Rather than raw probability, IPI incorporates vulnerable dependents: `IPI = Prob * (70 + 30 * VulnerabilityFactor)`. Households with infants and elderly get prioritized for immediate doorstep safety demos and refill subsidies."*
     - Point out the 3 human-readable reason codes dynamically generated under the gauge.
2. **Click on Tab 3: Outreach Triage & Priority Dispatch:**
   - *"In Tab 3, district administrators can filter by state and priority tier (Tier 1 to Tier 4) and immediately export the dispatch list for field teams."*
3. **Show Live Cloud Run Swagger UI (`https://c2c-app-685823552970.asia-south1.run.app/docs`):**
   - *"To make this model production-ready for state IT infrastructure, we deployed it live on Google Cloud Run with OpenAPI Swagger documentation."*
   - Show `GET /health`, `GET /metrics`, and `POST /predict`.
   - *"Any district tablet or state portal can call `POST /predict` or `POST /predict_short` with sub-100ms response times."*

---

## Phase 3: Q&A, Fairness Slices & Integrity (5:00 – 8:00)
> **Rubric D3: 7 Points + Rubric C1: 10 Points**  
> *Judge Expectation*: Deep understanding of pipeline, weaknesses, fairness slices across states, and honest negative results.

### Slide 7: Fairness Slices & Failure Modes
**Key Points to Explain:**
- *"**Fairness Audit (docs/error_analysis.md):** We evaluated performance across all 21 states. High agrarian states like UP (0.932 AUC) and Bihar (0.928 AUC) show clean signal separation. Kerala (0.871 AUC) is the most challenging due to high universal electrification, where subtle secondary fuel stacking occurs."*
- *"**Identified Failure Mode - 'Fuel Stacking':** Some affluent rural households own an LPG cylinder but continue to use biomass chulhas to boil large batches of cattle feed because crop residue is free. Our model occasionally predicts high clean fuel for them because their asset wealth is high. We documented this as our primary real-world failure mode."*
- *"**Honest Negative Result:** We experimented with Target Encoding on district IDs. While local validation showed a spurious +0.015 AUC improvement, it collapsed on unseen districts. We discarded it in favor of strictly invariant asset proxies."*

---

## Anticipated Judge Questions & Instant Winning Answers

| Question | Winning Answer |
| :--- | :--- |
| **Q1: Why not use a Deep Neural Network (MLP / TabNet)?** | *"On tabular survey data with 11,523 rows and missing values, GBDTs consistently outperform deep learning while running in under 4 seconds on 2 CPUs without GPU dependency, strictly honoring the evaluator resource sandbox."* |
| **Q2: Why not use stated monthly income as a primary feature?** | *"In rural household surveys, stated informal income has over 40% non-response or extreme underreporting due to fear of losing welfare benefits. Durable assets (refrigerator ownership, TV screen size, pucca roof) and electricity consumption are tamper-resistant proxies of true economic status."* |
| **Q3: How do you guarantee zero data leakage in your 5-fold CV?** | *"All preprocessing, frequency aggregations, and categorical mappings are fitted strictly on training folds inside our scikit-learn cross-validation loop. Zero validation fold information leaks into the transformer fit."* |
| **Q4: What if a household is off-grid with no electricity bill?** | *"Both XGBoost and LightGBM use learned missingness routing. When electricity features are NaN, the tree automatically branches to the off-grid node, which naturally increases the solid fuel probability without crashing."* |
| **Q5: What are your bonus claims?** | *"We claim all three bonus tracks: 1) Short 15-question model (+2 pts, 0.8961 AUC within 0.024 of full model), 2) Field-officer explanations (+2 pts, top_reasons column), and 3) Demographic fairness slice report in docs/error_analysis.md (+2 pts)."* |
| **Q6: Did you commit any data files over 5 MB?** | *"No. All data directories and submission files are untracked via `.gitignore`. Evaluator schema check passed 8 out of 8 checks cleanly."* |

---

## Presentation Checklist Before Pitching
- [x] Streamlit app running on `http://localhost:8501` (Tab 2 and Tab 3 pre-loaded).
- [x] Cloud Run Swagger UI open in a tab: `https://c2c-app-685823552970.asia-south1.run.app/docs`.
- [x] Presentation Deck open in a tab: `c:\Users\LENOVO\OneDrive\Desktop\C2C\presentation.html` (Press `F` or Fullscreen button).
- [x] Evaluator schema check verified: `python evaluator_schema_check.py` (8/8 PASS).
