from modules import regression_model_accuracy, plotActualVsPredicted, splitData, preProcessData, viewCorrelation, removeFeatures, find_outliers, pcaTraining, pcaTest, linearRegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Linear Regression w/o PCA
"""
# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove bad stuff from data
data_preprocessed = preProcessData(df)

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_preprocessed, test_size=0.3, random_state=15)

# Linear Regression

y_pred = linearRegressionModel(X_train, X_test, y_train, y_test)
# Plot of predicted values vs actual values
plotActualVsPredicted(y_test, y_pred)
# View how many  predictions were classified correctly within 30%
accuracy = regression_model_accuracy(y_test, y_pred, tolerance=0.3)
print(f"Percentage of predictions within 30% of actual values: {accuracy}%")
"""

# Polynomial Regression w/o PCA
"""
# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove bad stuff from data
data_preprocessed = preProcessData(df)

# Remove low-correlated features from data
data_clean = removeFeatures(data_preprocessed) 

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_clean, test_size=0.3, random_state=15)

# Find outliers in training set
outlier_indices = find_outliers(X_train)
# Remove outliers from the training data and labels
X_train_clean = X_train.drop(index=outlier_indices)
y_train_clean = y_train.drop(index=outlier_indices)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Assuming the necessary preprocessing functions have been defined and executed as per your description

# Generate Polynomial Features
degree = 2 # You can adjust this based on model performance and complexity
poly_features = PolynomialFeatures(degree=degree)

# Transform the training and testing data to include polynomial features
X_train_poly = poly_features.fit_transform(X_train_clean)
X_test_poly = poly_features.transform(X_test)

# Fit a Linear Regression Model
model = LinearRegression()
model.fit(X_train_poly, y_train_clean) # Note we're using the cleaned training data without outliers

# Make Predictions and Evaluate the Model
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculate and print the evaluation metrics
train_mse = mean_squared_error(y_train_clean, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train_clean, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse}, Train R^2: {train_r2}")
print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")
"""

# Ridge Regression w/o Data processing
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV


# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Assume preProcessData and splitData are predefined functions for preprocessing and splitting your dataset.
data_preprocessed = preProcessData(df)

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_preprocessed, test_size=0.3, random_state=15)

# List of alphas to try and find the best one through cross-validation
alphas = [1.0, 10.0, 15.0, 20.0]

# Initializing Ridge Regression model with cross-validation
ridge_reg_cv = RidgeCV(alphas=alphas, cv=5)  # cv=5 specifies a 5-fold cross-validation

# Fitting the Ridge Regression model with cross-validation to the training data
ridge_reg_cv.fit(X_train, y_train)

# The best alpha value after cross-validation
best_alpha = ridge_reg_cv.alpha_
print("Best alpha:", best_alpha)

# Predicting the test set results using the best model
y_pred = ridge_reg_cv.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Optional: Inspect the coefficients
print("Coefficients:", ridge_reg_cv.coef_)
"""

# Ridge Regression w/Data Processing
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV  # Corrected import here
import numpy as np

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove bad stuff from data
data_preprocessed = preProcessData(df)

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
train_pca, pca_optimal, scaler = pcaTraining(X_train_clean, 14, show_graph=False)
# Apply PCA transform to test data
test_pca = pcaTest(X_test, pca_optimal, scaler)

# Assuming `train_pca`, `test_pca`, `y_train_clean`, and `y_test` are already defined from your preprocessing and PCA

# Define a list of alpha values to explore
alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 10.0, 20.0]

# Initialize the Ridge Regression model with cross-validation
ridge_cv = RidgeCV(alphas=alphas, cv=5)  # Using 5-fold cross-validation

# Fit the model on the PCA-transformed training data
ridge_cv.fit(train_pca, y_train_clean)

# Identify the best alpha value after cross-validation
best_alpha = ridge_cv.alpha_
print("Optimal alpha value:", best_alpha)

# Predict using the Ridge model on the PCA-transformed test data
y_pred = ridge_cv.predict(test_pca)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Optional: You might want to compare these results with the Linear Regression baseline
# to assess the improvement from using Ridge Regression and the selected preprocessing steps.
"""

# Lasso Regression w/o PCA
"""
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Assume preProcessData and splitData are predefined functions for preprocessing and splitting your dataset.
data_preprocessed = preProcessData(df)

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_preprocessed, test_size=0.3, random_state=15)


# Initialize the Lasso regressor with an alpha value. 
# Alpha is the regularization strength; larger values specify stronger regularization.
lasso_reg = Lasso(alpha=0.01)

# Fit the Lasso model to the training data
lasso_reg.fit(X_train, y_train)

# Predict on the training and test sets
y_train_pred = lasso_reg.predict(X_train)
y_test_pred = lasso_reg.predict(X_test)

# Calculate and print the performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R^2: {train_r2}")
print(f"Test R^2: {test_r2}")
"""

# Lasso Regression w/PCA
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV  # Corrected import here
import numpy as np

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Remove bad stuff from data
data_preprocessed = preProcessData(df)

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
train_pca, pca_optimal, scaler = pcaTraining(X_train_clean, 14, show_graph=False)
# Apply PCA transform to test data
test_pca = pcaTest(X_test, pca_optimal, scaler)

# Initialize the RidgeCV regressor with default alphas or you can specify the alphas range
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13))

# Fit the RidgeCV model to the PCA-transformed training data
ridge_cv.fit(train_pca, y_train_clean)

# Predict on the PCA-transformed test data
y_test_pred = ridge_cv.predict(test_pca)

# Calculate performance metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test MSE: {test_mse}")
print(f"Test R^2: {test_r2}")
print(f"Optimal alpha: {ridge_cv.alpha_}")
"""

# Elastic Net w/o PCA
"""
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Assume preProcessData and splitData are predefined functions for preprocessing and splitting your dataset.
data_preprocessed = preProcessData(df)

# Split Data into train/test features
X_train, X_test, y_train, y_test = splitData(data_preprocessed, test_size=0.3, random_state=15)

# Define the model
elastic_net_model = ElasticNet(random_state=15)

# Define the parameter grid to search over
param_grid = {
    'alpha': [0.01, 0.05, 0.1, 0.2],
    'l1_ratio': [0.01, 0.05, 0.1, 0.2]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=elastic_net_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Predict on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {best_params}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
"""

# Neural Networks w/o PCA
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Assuming preProcessData and splitData are your custom functions

# Load up the data
df = pd.read_csv('household_energy_data.csv')

# Assume preProcessData and splitData are predefined functions for preprocessing and splitting your dataset.
data_preprocessed = preProcessData(df)

# Split Data into train/test features and labels
X_train, X_test, y_train, y_test = splitData(data_preprocessed, test_size=0.3, random_state=15)

# Convert to numpy arrays if they're not already (especially for pandas DataFrame/Series)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define a Dataset Class
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        # Ensure features and labels are numpy arrays
        self.features = np.array(features)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Ensure the output is always an np.ndarray
        feature = self.features[idx]
        label = self.labels[idx]
        
        # If the output is a scalar (numpy.float64), convert it to a 1D array
        if isinstance(feature, np.float64):
            feature = np.array([feature])
        if isinstance(label, np.float64):
            label = np.array([label])
            
        return torch.from_numpy(feature).float(), torch.from_numpy(label).float()

train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Neural Network Model remains the same
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out.view(-1, 1)  # Reshape output to have the shape [batch_size, 1]


# Determine input size dynamically
input_size = X_train.shape[1]
hidden_size = 50 # Example hidden size
output_size = 1

model = NeuralNet(input_size, hidden_size, output_size)

# 4. Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Ensure targets have the same dimensionality as outputs
        targets = targets.view(-1, 1) if len(targets.shape) == 1 else targets
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 5. Evaluate the Model
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        # Ensure targets have the same dimensionality as outputs
        targets = targets.view(-1, 1) if len(targets.shape) == 1 else targets
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader)}')
"""

# Decision Tree

# Random Forest

# Support Vector Regression
