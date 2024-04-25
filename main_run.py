from modules import k_fold_cv, regression_model_accuracy, plotActualVsPredicted, splitData, preProcessData, viewCorrelation, removeFeatures, find_outliers, pcaTraining, pcaTest, linearRegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# READ ME
# TO DOWNLOAD ALL LIBRARIES USED, RUN IN YOUR TERMINAL:
# pip install numpy scipy scikit-learn matplotlib seaborn pandas



# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove Infinite and NaN values from data
data_preprocessed = preProcessData(df)

# View correlation between features
viewCorrelation(data_preprocessed) # View correlation between features
# Remove low-correlated features from data
data_clean = removeFeatures(data_preprocessed) 

# Split Data into train/test sub-datasets.
X_train, X_test, y_train, y_test = splitData(data_clean, test_size=0.3, random_state=15)


# Find outliers in training set
outlier_indices = find_outliers(X_train)
# Remove outliers from the training data and their corresponding labels
X_train_clean = X_train.drop(index=outlier_indices)
y_train_clean = y_train.drop(index=outlier_indices)

# apply PCA to train data
train_pca, pca_optimal, scaler = pcaTraining(X_train_clean, 14, show_graph=True)
# Apply PCA transform to test data
test_pca = pcaTest(X_test, pca_optimal, scaler)

# Using a linear regression model:
y_pred = linearRegressionModel(train_pca, test_pca, y_train_clean, y_test)
# Plot of predicted values vs actual values
plotActualVsPredicted(y_test, y_pred)
# View how many  predictions were classified correctly within 30%
accuracy = regression_model_accuracy(y_test, y_pred, tolerance=0.3)
print(f"Percentage of predictions within 30% of actual values: {accuracy}%")


# Replace removal of rows w/ NaN/Inf values with KNN Imputation
# K-fold Cross Validation
# Replace Z-Score with Isolation Forest































