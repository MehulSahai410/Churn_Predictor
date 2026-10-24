# Customer Churn Predictor

This project predicts whether a bank customer is likely to leave the bank using machine learning. The main goal is to understand customer behaviour, identify patterns associated with churn, and compare different classification models to find a suitable prediction approach.

## About the Project

Customer churn is an important problem for banks because losing existing customers can affect revenue and long-term growth. This project uses customer information such as age, credit score, account balance, tenure, geography, and activity status to predict whether a customer is likely to leave.

I started with data exploration and preprocessing, built a Logistic Regression model as a baseline, and then developed an XGBoost model to compare their performance.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Joblib

## What I Did

- Explored the dataset to understand customer attributes and their distributions.
- Checked for missing values and examined numerical features for potential outliers.
- Removed unnecessary columns and separated the target variable, `Exited`.
- Used a stratified train-test split to maintain the target distribution.
- Applied MinMaxScaler to numerical features and OneHotEncoder to categorical features.
- Trained a Logistic Regression model with class weighting as the baseline.
- Built an XGBoost classification pipeline for comparison.
- Evaluated both models using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
- Compared the models using confusion matrices and ROC curves.
- Saved the trained XGBoost pipeline using Joblib.

## Models Used

**Logistic Regression:** Used as the baseline model to establish an initial performance benchmark.

**XGBoost:** Used to capture more complex relationships in customer data and compare its performance with the baseline model.

## Dataset

The project uses the `Customer-Churn-Records.csv` dataset. The target column is `Exited`, where:

- `0` means the customer stayed with the bank.
- `1` means the customer left the bank.

The input features include credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary, and card type.

## How to Run

1. Clone this repository.

   ```bash
  git clone https://github.com/MehulSahai410/Churn_Predictor.git
  cd Churn_Predictor
   ```

2. Install the required libraries.

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
   ```

3. Place `Customer-Churn-Records.csv` in the project directory.

4. Start Jupyter Notebook.

   ```bash
   jupyter notebook
   ```

5. Open `Churn-Predictor.ipynb` and run the cells in order.

After running the notebook, you can view the model evaluation results and the comparison plots. The trained XGBoost pipeline is saved as `customer_churn_xgb_pipeline.pkl`.

## Future Improvements

- Build a Streamlit interface for interactive churn predictions.
- Add feature importance and explainability using SHAP.
- Tune model hyperparameters to improve performance.
- Develop customer-level churn risk scores to support retention strategies.

## Conclusion

This project helped me understand the complete machine learning workflow, from exploratory data analysis and preprocessing to model training, evaluation, and saving a trained pipeline. Comparing Logistic Regression with XGBoost also provided a practical way to study the differences between linear and tree-based classification models.
