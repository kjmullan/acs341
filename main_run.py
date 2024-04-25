# READ ME
# TO DOWNLOAD ALL LIBRARIES USED, RUN IN YOUR TERMINAL:
# pip install numpy scipy scikit-learn matplotlib seaborn pandas

from modules import preProcessData, viewCorrelation, removeFeatures, find_outliers, pcaTraining, pcaTest, linearRegressionModel, plotActualVsPredicted
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Data preprocessing
data_preprocessed = preProcessData(df)

# Viewing correlation and feature removal
viewCorrelation(data_preprocessed)
data_clean = removeFeatures(data_preprocessed)

# Setting up K-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initializing lists to store metrics for each fold
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(data_clean):
    # Splitting data into train and test sets for the current fold
    X = data_clean.iloc[:, 1:]  # Features: all columns except the first
    y = data_clean.iloc[:, 0]   # Target: the first column

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Remove outliers from the training data
    outlier_indices = find_outliers(X_train)
    X_train_clean = X_train.drop(index=outlier_indices)
    y_train_clean = y_train.drop(index=outlier_indices)

    # Apply PCA to train data
    train_pca, pca_optimal, scaler = pcaTraining(X_train_clean, 14, show_graph=False)
    test_pca = pcaTest(X_test, pca_optimal, scaler)

    # Using a linear regression model
    y_pred = linearRegressionModel(train_pca, test_pca, y_train_clean, y_test)

    # Computing metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)

    # Plot actual vs predicted values
    plotActualVsPredicted(y_test, y_pred)

# Display average metrics across all folds
print(f"Average Mean Squared Error: {np.mean(mse_scores)}")
print(f"Average R^2 Score: {np.mean(r2_scores)}")


# Replace removal of rows w/ NaN/Inf values with KNN Imputation
# K-fold Cross Validation
# Replace Z-Score with Isolation Forest































