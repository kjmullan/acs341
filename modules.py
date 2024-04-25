import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd

# Split data into train and test data
def splitData(df, test_size, random_state):

    # Separate the dataset into features and target variable
    y = df.iloc[:, 0]  # Target variable is in the first column
    X = df.iloc[:, 1:]  # Features are all other columns

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Remove rows with infinite and NaN values
def preProcessData(data):
    # Replace 'Inf' and '-Inf' with 'NaN'
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any 'NaN' values (which now includes the original 'Inf'/'-Inf' values)
    data.dropna(inplace=True)

    # Applying one-hot encoding to the 'WeatherIcon' column

    data_encoded = data.drop(columns="WeatherIcon")
    # data_encoded = pd.get_dummies(data, columns=['WeatherIcon'])

    return data_encoded

### USING SEABORN 
def viewCorrelation(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()
### USING SEABORN

# Remove features based upon correlation within features.
def removeFeatures(data):
    # Based upon correlation and multicollinearity List of features to remove
    features_to_remove = [
        'apparentTemperature', 'Solar_kW_', 'dewPoint',
        'windBearing', 'humidity', 'Oven_kW_', 'pressure',
        'temperature', 'precipIntensity', 'GarageDoor_kW_',
        'visibility', 'Kettle_kW_', 'windSpeed', 'precipProbability'
    ]
    # Remove the specified features from the dataset
    data_reduced = data.drop(columns=features_to_remove)

    return data_reduced

# Remove those outside outlier scores
def find_outliers(data):
    # Z-Score normalization
    data_normalised = data.apply(zscore)
    # Calculate the sum of squared Z-scores for each data point
    outlier_scores = (data_normalised ** 2).sum(axis=1)
    # Determine a threshold for outlier scores, e.g., the 95th percentile
    threshold_outlier_score = outlier_scores.quantile(0.99)
    # Identify outlier indices
    outlier_indices = outlier_scores[outlier_scores > threshold_outlier_score].index

    return outlier_indices


def pcaTraining(train_data, n_components, show_graph=True):

    # Normalising data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(train_data)
    # Applying pca
    pca_optimal = PCA(n_components)
    data_pca_transformed = pca_optimal.fit_transform(data_scaled)

    # Calculate cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca_optimal.explained_variance_ratio_)

    if show_graph:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
        plt.title('Explained Variance Ratio by Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.grid(True)
        plt.show()

    return data_pca_transformed, pca_optimal, scaler


def pcaTest(test_data, pca_optimal, scaler):

    # Standardizing the test data using the fitted scaler
    test_data_scaled = scaler.transform(test_data)
    
    # Transforming the test data using the fitted PCA
    test_data_pca_transformed = pca_optimal.transform(test_data_scaled)

    return test_data_pca_transformed

def linearRegressionModel(train_pca, test_pca, y_train_clean, y_test):
    # Define the model
    model = LinearRegression()
    
    # Fit the model to your training data
    model.fit(train_pca, y_train_clean)
    
    # Predict the values of y using the test data
    y_pred = model.predict(test_pca)
    
    # Calculate and print the MSE/R^2
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    """
    # Accessing and printing the weights and intercept
    weights = model.coef_
    intercept = model.intercept_

    print("Weights: ", weights)
    print("Intercept: ", intercept)
    """

    return y_pred

def plotActualVsPredicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    # Scatter plot for actual vs predicted values
    sns.scatterplot(x=y_test, y=y_pred)
    
    # Calculate the limits for the plot
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    # Plotting the line for perfect predictions
    sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color='red', label='Ideal Predictions')
    
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

def regression_model_accuracy(y_test, y_pred, tolerance):
    """
    Evaluate the accuracy of a regression model.

    Parameters:
    - y_test: The actual values from the test dataset.
    - y_pred: The predicted values from the model.
    - tolerance: The percentage within which predictions are considered accurate.

    Returns:
    - The percentage of predictions that are within the specified tolerance of the actual values.
    """

    # Calculate the absolute error between predicted and actual values
    absolute_error = np.abs(y_pred - y_test)

    # Calculate the percentage error relative to the actual values
    percentage_error = absolute_error / y_test

    # Identify predictions within the specified tolerance
    accurate_predictions = percentage_error <= tolerance

    # Calculate the percentage of predictions that are accurate
    accuracy_percentage = np.mean(accurate_predictions) * 100

    return accuracy_percentage


