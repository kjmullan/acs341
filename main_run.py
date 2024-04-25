from modules import regression_model_accuracy, plotActualVsPredicted, splitData, preProcessData, viewCorrelation, removeFeatures, find_outliers, pcaTraining, pcaTest, linearRegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove bad stuff from data
data_preprocessed = preProcessData(df)

# View correlation between features
viewCorrelation(data_preprocessed) # View correlation between features
# Remove low-correlated features from data
data_clean = removeFeatures(data_preprocessed) 

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_clean, test_size=0.3, random_state=15)


# Find outliers in training set
outlier_indices = find_outliers(X_train)
# Remove outliers from the training data and labels
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


# See what the accuracy is without the data cleaning






























# Pyspark, Ray = Distributed Processing
# Tensorflow, Pytorch = Deep Learning
# Scikit Learn, Pyspark = Machine Learning
# Pandas, Pyspark = ETL
# Matplotlib, Plotly (Best), Seaborn = Data Visualisation

"""
import pandas as pd
import glob

# Find all files matching the pattern
files = glob.glob('household_energy_data.csv*')

# Optional: Sort the files if needed, this might help in selecting the most relevant file
files.sort()

# Check if any files were found
if files:
    # Read the first file from the list
    df = pd.read_csv(files[0])
    print("Data loaded from:", files[0])
else:
    print("No files found matching the pattern.")
"""
