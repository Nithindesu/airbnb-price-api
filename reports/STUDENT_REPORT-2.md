---
editor_options: 
  markdown: 
    wrap: 72
---

# AirBnB Price Prediction — Student Report Template

Venkata Nithin Desu

------------------------------------------------------------------------

## 1) Reproducibility + Submission

Checklist: - [ **TRUE** ] My notebook/script runs end-to-end from
`train.csv` + `test.csv` to `submission.csv` - [ **TRUE** ]
`submission.csv` has exactly two columns: `id,price` - [ **TRUE** ]
`submission.csv` has the correct number of rows and no missing ids - [
**TRUE** ] Random seed(s) are set and documented

Exact steps to reproduce (copy/paste commands or “Run all cells”
instructions):

**Run all cells sequentially.**

**The pipeline performs preprocessing, feature engineering, model
training, validation, and prediction.**

**The final predictions are saved as submission.csv in the correct
format.**

------------------------------------------------------------------------

## 2) Validation Design + Leakage Discussion

Validation setup (describe your split/CV, what you measure, and why):

**The dataset was split into training and validation sets using an 80/20
random split. A fixed random seed was used to ensure that results are
reproducible across runs.**

**Model performance was evaluated using Root Mean Squared Log Error
(RMSLE), which is the official metric used in the competition. RMSLE is
appropriate for this problem because Airbnb prices have a right-skewed
distribution, and the metric penalizes large relative prediction errors
rather than absolute differences.**

Assumptions you are making about the data / evaluation:

**Airbnb listing prices are right-skewed, with many inexpensive listings
and fewer high-priced listings.**

**Listings that can accommodate more guests or have more bedrooms are
likely to have higher prices.**

**Missing values in review-related features may indicate that a listing
has not yet received reviews, rather than true missing data.**

**These assumptions guided the preprocessing and feature engineering
steps.**

Leakage risks you considered and how you avoided them: -
`weekly_price`/`monthly_price` excluded as features -
Encoders/statistics fit on training folds only - Any other leakage
risks:

**weekly_price and monthly_price were excluded from the feature set,
because these columns represent alternative pricing labels and would
directly leak information about the target variable.**

**Feature standardization was performed using only the training data,
and the same statistics were applied to the validation and test sets.**

**Date-based features were created using listing timestamps only,
without referencing the target variable.**

------------------------------------------------------------------------

## 3) EDA + Experiments (at least 2)

What you learned from EDA (missingness, distributions, outliers, feature
issues):

**The distribution of listing prices is right-skewed, with most listings
priced between \$50–\$300.**

**Listings that accommodate more guests tend to have higher prices.**

**Review score features have weak correlation with price compared to
capacity-related features such as accommodates and bedrooms.**

**Some features contain missing values which were handled during
preprocessing.**

Experiments / ablations (minimum 2). Include results:

| Experiment | Change | Validation RMSLE | Notes / Interpretation |
|----|----|---:|----|
| Baseline | Ridge + basic features | 0.623697 | Linear model baseline |
| Exp 1 | XGBoost | 0.293480 | Captures nonlinear relationships |
| Exp 2 | LightGBM | 0.296703 | Similar performance to XGBoost |

------------------------------------------------------------------------

## 4) Final Model + Performance

Final model description (model type, key hyperparameters, key features):

**The final model selected was XGBoost, a gradient boosting algorithm
that performs well on tabular datasets.**

**Key hyperparameters used:**

**n_estimators = 4000**

**learning_rate = 0.03**

**max_depth = 8**

**subsample = 0.85**

**colsample_bytree = 0.85**

**Important features include:**

**accommodates**

**bedrooms**

**bathrooms**

**zipcode**

**availability_365**

**engineered date features (host_tenure_days, days_since_first_review)**

Did you use embeddings (`airbnb-use-embeddings.csv`)? If yes, how
(sample/SVD, dimensionality, etc.):

**NO We haven't Used embeddings**

Validation RMSLE (your best):

**XGboost model And The Validation RMSLE IS : 0.293480**

Ridge baseline RMSLE (from `starter_baseline.ipynb`): **0.623697**

Class leaderboard RMSLE (your final submission):

Did you beat the ridge baseline on the leaderboard? (yes/no):

------------------------------------------------------------------------

## 5) What Helped Most / Next Steps

Biggest wins:

**Feature engineering from date columns**

**Cleaning money and percentage fields**

**Using gradient boosting models such as XGBoost and LightGBM**

If you had more time, what would you try next?

**From the current modeling approach, we expect the model to capture
important relationships between listing attributes and price,
particularly factors such as accommodation capacity, location, and
property characteristics. Tree-based models like XGBoost and LightGBM
help capture nonlinear relationships and improve prediction accuracy
compared to the baseline linear model.**

**For further improvements, additional feature engineering could be
explored, especially stronger location-based features. Better encoding
methods for categorical variables such as zipcode may also improve
performance. Explore additional feature interactions such as ratios
between bedrooms, bathrooms, and accommodates. More extensive
hyperparameter tuning and experimenting with additional feature
interactions could further reduce the RMSLE score.**
