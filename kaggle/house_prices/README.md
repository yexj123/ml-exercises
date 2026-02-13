# Ames Housing Price Prediction 🏠

A machine learning project to predict house prices in Ames, Iowa, using the [Kaggle Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

This project was built to apply the concepts from **Chapter 2** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* by Aurélien Géron.

## 🏆 Results

- **Kaggle Public Score (RMSLE):** `0.13426`
- **Leaderboard Position:** `1914` Up to Date 13/02/2026 12:55 Rome time
- **Model Used:** Support Vector Regressor (SVR) with Lasso Feature Selection

## 🛠️ Key Techniques Used

1.  **Data Cleaning:**
    - Handled missing data differently for categorical features (e.g., `NaN` in 'Garage' means 'No Garage', not missing data).
    - Detected and fixed missing values using `SimpleImputer`.

2.  **Feature Engineering:**
    - **Log-Transformation:** Applied `np.log1p` to skewed features (like `LotArea`) to normalize distributions.
    - **Target Transformation:** Used `TransformedTargetRegressor` to log-transform the target (`SalePrice`), ensuring the model optimizes for RMSLE (Kaggle's metric).

3.  **Pipeline Construction:**
    - Built a `Scikit-Learn` pipeline handling numerical scaling (`StandardScaler`), One-Hot Encoding, and Ordinal Encoding simultaneously.
    - Integrated `SelectFromModel` with **Lasso** to automatically select the most important features from the 80+ available columns.

4.  **Hyperparameter Tuning:**
    - Used `RandomizedSearchCV` to simultaneously tune the Feature Selector (`alpha`) and the Regressor (`C`, `epsilon`) to find the optimal balance.

## 📚 Acknowledgments

- **Aurélien Géron** for the incredible "End-to-End Machine Learning Project" walkthrough in Chapter 2.
- **Kaggle** for hosting the dataset and competition.