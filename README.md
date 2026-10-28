# Customer Churn Predictor

An end-to-end machine learning project that predicts whether a bank customer is likely to leave using customer demographic, account, and behavioural data. The project compares Logistic Regression and XGBoost, optimizes the churn decision threshold using cross-validation, and provides an interactive Streamlit application for individual customer risk assessment.

## Overview

Customer churn can affect a bank's revenue and long-term growth. However, retention campaigns also have costs, making it important to identify customers who are genuinely at risk of leaving.

This project develops a machine learning pipeline to predict customer churn while addressing the class imbalance in the dataset. Instead of relying exclusively on accuracy or the default classification threshold of 0.50, it focuses on precision, recall, F1-score, and ROC-AUC to evaluate model performance.

The workflow includes exploratory data analysis (EDA), preprocessing, feature engineering, model comparison, cross-validated threshold selection, and deployment through Streamlit.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook

## Model Performance

The dataset was divided using a stratified 75/25 train-test split. The test set contains 2,500 customers, including 509 customers who churned.

### Model Comparison

| Model | ROC-AUC |
|---|---:|
| Logistic Regression (class-weighted baseline) | 0.7839 |
| **XGBoost pipeline** | **0.8752** |

XGBoost achieved a higher ROC-AUC than the Logistic Regression baseline, indicating stronger overall discrimination between customers who churn and those who remain.

### XGBoost Performance at the Selected Threshold

The classification threshold was adjusted from the default 0.50 to **0.35** to improve churn detection.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8752 |
| Churn Precision | 0.66 |
| Churn Recall | 0.61 |
| Churn F1-score | 0.63 |
| Customers Flagged for Retention | 475 / 2,500 |

At this threshold, approximately 66% of customers flagged by the model actually churned in the test set. The model identified 312 of the 509 actual churners.

### Confusion Matrix

| Actual Class | Predicted Stayed | Predicted Churned |
|---|---:|---:|
| Stayed | 1,828 | 163 |
| Churned | 197 | 312 |

The model correctly classified 2,140 out of 2,500 customers, corresponding to **85.6% accuracy** at the selected threshold.

- **True Negatives (1,828):** Customers who stayed and were correctly identified.
- **False Positives (163):** Customers flagged for churn who stayed.
- **False Negatives (197):** Customers who churned but were missed.
- **True Positives (312):** Customers who churned and were correctly identified.

These results illustrate the trade-off between identifying more potential churners and contacting customers who may not actually leave.

### Decision Threshold Comparison

| Threshold | Precision | Recall | F1-score | Customers Flagged |
|---:|---:|---:|---:|---:|
| 0.30 | 0.60 | 0.66 | 0.63 | 553 |
| **0.35** | **0.66** | **0.61** | **0.63** | **475** |
| 0.40 | 0.70 | 0.58 | 0.63 | 423 |
| 0.50 | 0.76 | 0.50 | 0.60 | 333 |

A lower threshold increases churn recall but generally produces more false positives. A higher threshold reduces unnecessary retention outreach but misses more customers who eventually churn.

The selected threshold of 0.35 offers a practical balance between churn detection and the number of customers flagged for intervention.

## Threshold Selection with Cross-Validation

The default classification threshold of 0.50 is not necessarily optimal for customer retention. To select a more suitable threshold without using test-set information during tuning, the following approach was implemented:

1. Generated out-of-fold churn probabilities on the training set using five-fold stratified cross-validation.
2. Refit the complete preprocessing and modelling pipeline within each fold to prevent preprocessing leakage.
3. Evaluated precision, recall, and F1-score across candidate thresholds from 0.05 to 0.95.
4. Compared candidate thresholds using cross-validated results.
5. Selected a threshold of 0.35 based on the trade-off between churn detection and the number of customers flagged.
6. Evaluated the selected threshold on the held-out test set for the final performance report.

The cross-validation results showed that F1-score was relatively stable around thresholds of 0.30 and 0.35. The threshold of 0.35 was chosen to reduce the number of customers flagged compared with 0.30 while maintaining a similar F1-score.

Accuracy is reported as a secondary metric because a classifier that predicts every customer will stay can already achieve approximately 80% accuracy on a dataset with this class distribution.

## Streamlit Application

The project includes an interactive Streamlit application that allows users to enter individual customer details and obtain a churn risk assessment.

The application includes:

- **Customer Input:** Demographic, account, and engagement-related information.
- **Churn Risk Score:** A visual representation of the model's predicted churn score.
- **Risk Classification:** Risk categories based on the selected decision threshold.
- **Retention Indicators:** Relevant customer characteristics, such as inactivity or holding only one bank product.
- **Threshold-Based Decision:** Customers with predicted churn scores of 0.35 or higher are flagged for potential retention action.

The app applies the saved preprocessing and XGBoost pipeline to customer inputs and uses the selected threshold to determine the predicted class.

### Run the Application

```bash
streamlit run app.py
```

## What I Implemented

- Conducted exploratory data analysis to understand customer distributions and relationships.
- Investigated missing values and potential numerical outliers.
- Removed identifier columns and the `Complain` feature, then separated the target variable `Exited`.
- Used a stratified train-test split to preserve the target-class distribution.
- Built a Scikit-learn `ColumnTransformer` for numerical and categorical preprocessing.
- Applied `MinMaxScaler` to numerical features and `OneHotEncoder` to categorical features.
- Trained a class-weighted Logistic Regression model as the baseline.
- Developed an XGBoost classification pipeline to capture nonlinear relationships and feature interactions.
- Compared models using precision, recall, F1-score, confusion matrices, and ROC-AUC.
- Generated out-of-fold probabilities using five-fold stratified cross-validation.
- Tuned the classification threshold to balance churn detection and retention outreach.
- Saved the trained XGBoost pipeline using Joblib.
- Developed a Streamlit application for interactive customer scoring and risk assessment.

## Dataset

**Dataset:** `Customer-Churn-Records.csv` — Kaggle Bank Customer Churn dataset.

The target variable is `Exited`:

- `0` — The customer stayed with the bank.
- `1` — The customer left the bank.

Input features include:

- Credit score
- Geography and gender
- Age and tenure
- Account balance
- Number of bank products
- Credit card status
- Active membership status
- Estimated salary
- Satisfaction score
- Card type
- Points earned

Identifier columns and the `Complain` feature were excluded during preprocessing.

## Project Structure

```text
Churn_Predictor/
├── app.py
├── Churn-Predictor.ipynb
├── customer_churn_xgb_pipeline.pkl
├── Customer-Churn-Records.csv
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application for customer churn prediction.
- `Churn-Predictor.ipynb` — EDA, preprocessing, model training, evaluation, and threshold tuning.
- `customer_churn_xgb_pipeline.pkl` — Serialized preprocessing and XGBoost pipeline.
- `Customer-Churn-Records.csv` — Dataset used for model development.
- `requirements.txt` — Project dependencies.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/MehulSahai410/Churn_Predictor.git
cd Churn_Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Explore the notebook

```bash
jupyter notebook
```

Open `Churn-Predictor.ipynb` and execute the cells in order to explore the dataset, train the models, evaluate their performance, and reproduce the threshold-selection workflow.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Ensure the saved pipeline and any required application files are available in the expected project paths.

## Limitations

- **Threshold selection depends on business objectives:** A threshold of 0.35 may not be optimal for every bank or retention campaign.
- **Single train-test split:** Reported test metrics may vary across different data splits.
- **Imbalanced target:** Churn recall remains imperfect, meaning some customers who leave will not be identified.
- **Probability calibration:** The model's predicted scores have not been calibrated. They should not automatically be interpreted as perfectly calibrated real-world churn probabilities.
- **No demonstrated financial impact:** The project does not yet quantify actual retention savings, campaign costs, or the long-term value of retained customers.
- **Model explainability:** More detailed feature-level and individual prediction explanations could improve transparency.

## Future Improvements

- Integrate SHAP to explain individual predictions and global feature importance.
- Tune XGBoost hyperparameters using cross-validation.
- Calibrate predicted probabilities and evaluate PR-AUC.
- Select thresholds using explicit retention costs and customer lifetime value assumptions.
- Add model monitoring and drift detection.
- Expand the Streamlit dashboard with aggregate churn trends and customer-segment analysis.
- Deploy the application on Streamlit Community Cloud.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for bank customer churn prediction, combining exploratory analysis, preprocessing, model comparison, cross-validated threshold selection, model persistence, and an interactive Streamlit application.

The XGBoost pipeline achieved a ROC-AUC of **0.8752**. At the selected threshold of 0.35, it achieved a churn precision of 0.66 and recall of 0.61 on the held-out test set.

The main takeaway is that effective churn prediction involves more than maximizing accuracy. Selecting an appropriate decision threshold allows a business to balance missed churners against the cost of unnecessary retention outreach, providing a foundation for more targeted, data-driven customer retention strategies.
