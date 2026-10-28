# Customer Churn Predictor

An end-to-end machine learning project that predicts whether a bank customer is likely to leave the bank using customer demographic, account, and behavioural data. The project compares Logistic Regression and XGBoost to evaluate their effectiveness in identifying potential customer churn.

## About the Project

Customer churn is a significant challenge for banks, as losing existing customers can affect revenue and long-term growth. This project explores customer behaviour, identifies patterns associated with churn, and develops classification models to predict whether a customer will leave the bank.

The workflow includes exploratory data analysis, data preprocessing, feature engineering, model training, performance evaluation, and model serialization. Logistic Regression serves as the baseline model, while XGBoost is used to capture more complex patterns in the data.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Persistence:** Joblib

## Key Results — XGBoost

The XGBoost model achieved **86% classification accuracy** and a **ROC-AUC score of 0.8752**, demonstrating its ability to distinguish between customers who stay and those who leave.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 86% |
| ROC-AUC | **0.8752** |
| Churn Precision | 0.76 |
| Churn Recall | 0.50 |
| Churn F1-Score | 0.60 |

**Key observations:**

- **Strong overall performance:** The model achieved 86% accuracy on 2,500 test samples.
- **Good discrimination capability:** A ROC-AUC of 0.8752 indicates that the model can distinguish between churned and retained customers substantially better than random ranking.
- **Reliable positive predictions:** A churn precision of 0.76 means that 76% of the customers predicted to churn actually churned in the test set.
- **Opportunity for improvement:** A churn recall of 0.50 indicates that the model identified approximately half of the actual churned customers, leaving room for improvement in customer retention applications.

## Confusion Matrix

The following confusion matrix shows the XGBoost model's predictions on the test dataset.

| Actual Class | Predicted Stayed | Predicted Churned |
|---|---:|---:|
| Stayed | 1,910 | 81 |
| Churned | 257 | 252 |

- **1,910 true negatives:** Customers who stayed and were correctly classified.
- **81 false positives:** Customers predicted to churn who actually stayed.
- **257 false negatives:** Customers who churned but were incorrectly classified as staying.
- **252 true positives:** Customers who churned and were correctly identified.

The model correctly classified **2,162 out of 2,500 customers**, resulting in 86.48% accuracy before rounding.

## Classification Report

```text
              precision    recall  f1-score   support

Stayed            0.88      0.96      0.92      1991
Churned           0.76      0.50      0.60       509

accuracy                              0.86      2500
macro avg         0.82      0.73      0.76      2500
weighted avg      0.86      0.86      0.85      2500
```

The model performs particularly well at identifying customers who stay with the bank. However, detecting customers who are likely to leave remains more challenging. This distinction is important because a churn prediction system should not only maximize overall accuracy but also identify customers who may benefit from retention efforts.

## What I Did

- Performed exploratory data analysis to understand customer attributes, distributions, and relationships.
- Checked missing values and examined numerical features for potential outliers.
- Removed unnecessary columns and separated the target variable, `Exited`.
- Used a stratified train-test split to preserve the target-class distribution.
- Applied `MinMaxScaler` to numerical features and `OneHotEncoder` to categorical features.
- Trained a class-weighted Logistic Regression model as the baseline.
- Developed an XGBoost classification pipeline for comparison.
- Evaluated both models using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
- Compared classification performance and visualized results using confusion matrices and ROC curves.
- Saved the trained XGBoost pipeline using Joblib for reuse.

## Models Used

### 1. Logistic Regression

Used as the baseline classification model to establish a performance benchmark. Class weighting was applied to account for the imbalance between retained and churned customers.

### 2. XGBoost

Used as the advanced classification model to capture nonlinear relationships and interactions between customer features. Its performance was evaluated against the Logistic Regression baseline using multiple classification metrics.

**Final XGBoost results:** 86% accuracy, 0.8752 ROC-AUC, and 0.60 F1-score for the churned class.

## Dataset

The project uses the `Customer-Churn-Records.csv` dataset.

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
- Card type

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/MehulSahai410/Churn_Predictor.git
cd Churn_Predictor
```

**2. Install the required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

**3. Add the dataset**

Place `Customer-Churn-Records.csv` in the project directory.

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

**5. Run the notebook**

Open `Churn-Predictor.ipynb` and execute the cells in order.

After execution, you can inspect the model evaluation metrics and comparison plots. The trained XGBoost pipeline is saved as `customer_churn_xgb_pipeline.pkl`.

## Future Improvements

- Develop a Streamlit dashboard for interactive customer churn predictions.
- Integrate SHAP to explain individual predictions and identify influential features.
- Tune XGBoost hyperparameters using cross-validation.
- Optimize the classification threshold to improve churn recall while controlling false positives.
- Develop customer-level churn risk scores to support targeted retention strategies.
- Evaluate precision-recall curves and business-oriented metrics to better reflect the cost of missed churn cases.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for bank customer churn prediction, covering exploratory data analysis, preprocessing, model development, evaluation, and model persistence.

The XGBoost model achieved **86% accuracy and a ROC-AUC of 0.8752**, with a churn precision of 0.76. The evaluation also revealed that churn recall remains an important area for improvement, highlighting the trade-off between overall classification performance and identifying customers at risk of leaving.

The project provides a practical foundation for developing explainable, data-driven customer retention solutions.
