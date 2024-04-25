from modules import regression_model_accuracy, plotActualVsPredicted, splitData, preProcessData, viewCorrelation, removeFeatures, find_outliers, pcaTraining, pcaTest, linearRegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import KFold

# READ ME
# TO DOWNLOAD ALL LIBRARIES USED, RUN IN YOUR TERMINAL:
# pip install numpy scipy scikit-learn matplotlib seaborn pandas

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Data preprocessing
data_preprocessed = preProcessData(df)

# Viewing correlation and feature removal
viewCorrelation(data_preprocessed)
data_clean = removeFeatures(data_preprocessed)

# Setting up K-Fold Cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# Initializing lists to store metrics for each fold
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(data_clean):
    # Splitting data into train and test sets for the current fold
    train_data, test_data = data_clean.iloc[train_index], data_clean.iloc[test_index]

    # Remove outliers from the training data
    outlier_indices = find_outliers(train_data)
    train_data_clean = train_data.drop(index=outlier_indices)

    # Apply PCA to train data
    train_pca, pca_optimal, scaler = pcaTraining(train_data_clean, 14, show_graph=False)
    test_pca = pcaTest(test_data, pca_optimal, scaler)

    # Using a linear regression model
    y_train_clean = train_data_clean.iloc[:, 0]
    y_test = test_data.iloc[:, 0]
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