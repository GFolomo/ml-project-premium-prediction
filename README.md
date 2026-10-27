# Health Insurance Premium Prediction

An end-to-end machine learning project that predicts an applicant's annual health insurance premium from demographic, lifestyle, and medical-history inputs, deployed as an interactive web app for underwriters.

**[Live demo](https://gires-ml-project-premium-prediction.streamlit.app/)** (hosted on Streamlit Community Cloud; if the app is asleep, click "Wake up" and allow about 30 seconds)

| App | Demo video |
|---|---|
| ![App screenshot](assets/app_screenshot.png) | [Watch the walkthrough](assets/app_demo.mp4) |

## Key result

A single global model scored 98.1% accuracy but still mispriced about 30% of customers by more than 10%. Error analysis traced the problem to applicants aged 25 and under, so I segmented by age and trained a separate model for each group. Both segments now beat the project's targets:

| Segment | Best model | Accuracy (R²) | RMSE | Predictions off by more than 10% |
|---|---|---|---|---|
| Young (25 and under) | Linear Regression | 98.9% | 292.81 | 2.1% |
| Rest (over 25) | XGBoost | 99.8% | 311.15 | 0.3% |

## Project background

The project is built against a case-study scope of work (SOW): As a Data Scientist/ML Engineer I was engaged by an insurer (Shield Insurance) to build a model that estimates health insurance premiums from factors such as age, smoking habits, BMI, and medical history.

The SOW defines two phases:

1. **Phase 1 (MVP):** build and deploy a predictive model with a Streamlit application. *This repository covers Phase 1.*
2. **Phase 2:** infrastructure for straight-through processing (STP) of insurance quotes (out of scope here).

### Phase 1 objectives

- A high-accuracy model: **at least 97% accuracy**, and **at least 95% of predictions must have a percentage error below 10%** versus the actual premium
- A model deployed in the cloud so an underwriter can use it from anywhere
- An interactive Streamlit app for entering applicant details and viewing the predicted premium

### How the deliverables map to this repo

| SOW deliverable | Where it lives |
|---|---|
| Trained model | `artifacts/` (models and scalers saved with joblib) |
| Model development notebook | `notebooks/MLProject_Health-Insurance-Premium_Regression.ipynb` |
| Streamlit app | `main.py` |
| Prediction logic | `prediction_helper.py` |
| Deployed model | [Live demo](https://gires-ml-project-premium-prediction.streamlit.app/) |
| Documentation | This README |
| Training material | `assets/app_demo.mp4` (app walkthrough) |

## How the problem was approached, stage by stage

### 1. Business understanding

Underwriters need a fast, consistent premium estimate. The SOW sets two measurable targets: at least 97% accuracy, and at least 95% of predictions within 10% of the actual premium. The second target is the harder one, because it means almost every individual customer is priced fairly, not just the average one. I used both as my success criteria throughout.

### 2. Data understanding

The dataset has **50,000 records and 13 columns**: age, gender, region, marital status, number of dependants, BMI category, smoking status, employment status, income level, income (in lakhs), medical history, insurance plan, and the target, annual premium amount (range 3,501 to 43,471).

### 3. Data cleaning

| Issue found | Action and reasoning |
|---|---|
| Inconsistent column names | Standardized to lowercase snake_case |
| Missing values in `smoking_status` (11), `employment_status` (2), and `income_level` (13) | Dropped 24 rows (under 0.05% of the data), so imputation would add risk for no benefit |
| Duplicate rows | None found |
| Negative `number_of_dependants` (72 rows, values -3 and -1) | Converted to absolute values, assuming a sign entry error, to avoid losing 72 records |
| Impossible ages (values from 124 to 356) | Dropped 58 rows; the maximum valid age became 72 |
| Extreme `income_lakhs` values (max 930) | The standard IQR rule would have removed 3,559 rows (about 7%), which is too much real data. I used the 99.9th percentile (100 lakhs) instead and removed only 10 rows |
| `smoking_status` had 6 labels for 3 meanings ("Smoking=0", "Does Not Smoke", "Not Smoking") | Merged into "No Smoking" |

The cleaned dataset has **49,908 rows**.

### 4. Exploratory data analysis

Using **matplotlib and seaborn**, EDA covered both numeric and categorical columns:

- **Numeric:** box plots to find outliers, histograms with density curves for distributions, and scatter plots of age, income, and dependants against the premium
- **Categorical:** percentage distribution bar charts for every categorical column, plus a cross-tabulation and heatmap of income level against insurance plan to see how the two relate

### 5. Feature engineering

- **Medical risk score:** the `medical_history` column holds up to two conditions (e.g. "Diabetes & Heart disease"). I split it into two columns, assigned a risk weight per condition (heart disease 8, diabetes 6, high blood pressure 6, thyroid 5, none 0), summed them, and min-max normalized the result to a 0-1 `normalized_risk_score`
- **Ordinal encoding** for ordered categories: insurance plan (Bronze 1, Silver 2, Gold 3) and income level (four bands, 1 to 4)
- **One-hot encoding** (`drop_first=True` to avoid the dummy variable trap) for unordered categories: gender, region, marital status, BMI category, smoking status, and employment status
- **Min-max scaling** of the numeric and ordinal features

### 6. Feature selection and multicollinearity

I used a **correlation heatmap** and the **Variance Inflation Factor (VIF)** (`variance_inflation_factor` from `statsmodels`) to find features that carry overlapping information.

- `income_level` (VIF 12.5) and `income_lakhs` (VIF 11.2) were highly collinear, which makes sense since one is a banded version of the other
- I dropped `income_level` and kept the more precise `income_lakhs`. Its VIF fell to **2.5**, and every remaining feature is below 5
- This matters most for linear models, where collinear features make coefficients unstable and hard to interpret

### 7. Modeling and evaluation (before segmentation)

First I trained on all age groups together, with a 70/30 train/test split (34,935 training and 14,973 test rows). I compared a linear baseline, a regularized linear model, and a tree ensemble:

| Model | Accuracy (R²) | RMSE | Predictions off by more than 10% |
|---|---|---|---|
| Linear Regression | 92.8% | 2,272.80 | n/a |
| Ridge Regression | 92.8% | 2,272.81 | n/a |
| XGBoost (default settings) | 98.1% | 1,169.41 | n/a |
| XGBoost (tuned with RandomizedSearchCV) | 98.1% | n/a | about 30% |

XGBoost clearly won on accuracy. Tuning with `RandomizedSearchCV` (3-fold cross-validation, scored on R²) gave the same accuracy, so the default was already near-optimal. This tuned XGBoost was the best global model.

### 8. Error analysis: where the model failed

An accuracy of 98.1% looked like a pass, but the SOW's second target is stricter, so I analyzed the individual errors:

- **About 30% of test predictions (4,487 of 14,973) were off by more than 10%**, far from the required maximum of 5%
- **549 customers were over- or under-charged by more than 50%**
- A root-cause analysis (comparing the features of badly predicted customers against everyone else, after reversing the scaling to read real values) showed the errors were concentrated in **young customers on the Bronze plan**: their mean age was about 22, and **97% were aged 25 or under**
- For these customers the actual premiums were as low as about 3,500, but the model predicted about 6,500 or more, so they were being overcharged

A single global model was averaging across two groups that behave differently, so it was good overall but unfair to the youngest applicants.

### 9. The fix: segmented modeling

Based on that finding, I split applicants at age 25 and trained and compared Linear Regression, Ridge, and XGBoost separately on each segment.

**Young segment (25 and under)**

| Model | Accuracy (R²) | RMSE | Predictions off by more than 10% |
|---|---|---|---|
| Linear Regression (selected) | 98.9% | 292.81 | 2.1% |
| Ridge Regression | 98.8% | 292.78 | n/a |
| XGBoost | 98.8% | 305.47 | n/a |

A simple linear model matched or beat XGBoost here, so I chose it: it is easier to explain to underwriters and cheaper to maintain.

**Rest segment (over 25)**

| Model | Accuracy (R²) | RMSE | Predictions off by more than 10% |
|---|---|---|---|
| Linear Regression | 95.4% | 1,477.78 | n/a |
| Ridge Regression | 95.4% | 1,477.77 | n/a |
| XGBoost, default settings (selected) | 99.8% | 311.15 | 0.3% |
| XGBoost, tuned with RandomizedSearchCV | 99.8% | n/a | n/a |

For older applicants the linear models were noticeably weaker (95.4%), so the non-linear XGBoost model was the clear choice. Tuning did not change its accuracy.

The segmented models use 18 features, adding `genetical_risk` to the 17 used in the global model after additional data was requested from the client.

### 10. Deployment

![Prediction routing](assets/prediction_routing.png)

1. The underwriter enters the applicant's details in the Streamlit form (age, dependants, income, insurance plan, gender, region, marital status, BMI category, smoking status, employment status, and medical history).
2. `prediction_helper.py` converts the inputs into the model's features: one-hot encoding, the normalized medical risk score, and so on.
3. The app routes by age to the young or rest model.
4. Numeric features are scaled with that segment's own saved scaler.
5. The model returns the predicted premium, shown instantly.

The trained models and scalers are saved with joblib in `artifacts/`, and the app is deployed on Streamlit Community Cloud.

## Results against the acceptance criteria

| Requirement | Target | Global model | Young segment | Rest segment |
|---|---|---|---|---|
| Accuracy | 97% or higher | 98.1% (met) | 98.9% (met) | 99.8% (met) |
| Predictions within 10% error | 95% or higher | about 70% (not met) | 97.9% (met) | 99.7% (met) |

Segmenting by age turned a model that failed the second requirement into one that meets both in each group. Because each segment is above 95%, the combined result is too.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, NumPy |
| Visualization and EDA | matplotlib, seaborn |
| Statistical checks | statsmodels (`variance_inflation_factor`), correlation analysis |
| Modeling | scikit-learn (Linear Regression, Ridge, MinMaxScaler, RandomizedSearchCV), XGBoost |
| Model persistence | joblib |
| App and deployment | Streamlit, Streamlit Community Cloud |
| Version control | Git, GitHub |

## Repository structure

```
├── artifacts/              # Saved models and scalers (.joblib)
├── assets/                 # README media
│   ├── app_screenshot.png      # App screenshot
│   ├── app_demo.mp4            # App demo video
│   └── prediction_routing.png  # Prediction routing screenshot
├── .gitignore
├── LICENSE                 # Apache-2.0
├── README.md
├── main.py                 # Streamlit UI
├── prediction_helper.py    # Preprocessing, segment routing, prediction
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/GFolomo/ml-project-premium-prediction.git
cd ml-project-premium-prediction
pip install -r requirements.txt
streamlit run main.py
```

## Limitations and next steps

- **Phase 2 of the SOW:** straight-through processing (STP) so quotes can be generated end to end without manual handling
- Fit scalers on training data only, inside a scikit-learn `Pipeline`, to rule out data leakage and simplify deployment
- Smoother handling of the age 25/26 segment boundary, since applicants switch models at that point
- Cross-validation of the final segmented models, and monitoring of prediction drift and error per segment
- The risk weights per medical condition are assigned judgments, and could be replaced with weights learned from claims data

## Author

**Gires Folomo**, Data Scientist, Johannesburg, South Africa
[https://gires-ml-project-premium-prediction.streamlit.app/]| [GitHub](https://github.com/GFolomo)
