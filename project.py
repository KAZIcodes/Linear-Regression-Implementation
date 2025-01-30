import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# PHASE 1: Data Analysis

# Load the dataset
data_path = './DataSet.csv'
data = pd.read_csv(data_path)

# Display basic dataset information
print("Dataset Info:\n")
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print(f"\nMissing Values in Each Column:\n{missing_values}")

# Check for outliers (using boxplots for visualization)
plt.figure(figsize=(12, 8))
for i, col in enumerate(data.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=data[col])
    plt.title(f"Box Plot - {col}")
plt.tight_layout()
plt.show()

# Visualize distributions of numerical features using Histograms
plt.figure(figsize=(12, 8))
data.select_dtypes(include=['float64', 'int64']).hist(bins=20, figsize=(12, 8), edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.show()

# Calculate basic statistics
stats_summary = {}
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    stats_summary[col] = {
        "Mean": data[col].mean(),
        "Median": data[col].median(),
        "Variance": data[col].var(),
        "Standard Deviation": data[col].std()
    }

print("\nStatistical Summary for Numerical Columns:")
for col, stats in stats_summary.items():
    print(f"\nColumn: {col}")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value}")

# Explore relationships between variables
sns.pairplot(data, vars=["age", "bmi", "children", "charges"], hue="smoker", palette="coolwarm")
plt.suptitle("Scatter Plots with 'Smoker' Hue", y=1.02, fontsize=16)
plt.show()

# Calculate and visualize the correlation matrix
correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features", fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data['bmi'], kde=True, bins=30)
plt.title("Distribution of BMI", fontsize=16)
plt.show()

# PHASE 2: Preprocess the data

# Step 1: Handle missing values
data_cleaned = data.dropna()

# Step 2: Encode categorical variables
data_encoded = pd.get_dummies(data_cleaned, columns=["sex", "smoker", "region"], drop_first=True)
data_encoded = data_encoded.astype(float)

# Step 3: Add new features
data_encoded['age_bmi_interaction'] = data_encoded['age'] * data_encoded['bmi']
data_encoded['smoker_bmi_interaction'] = data_encoded['smoker_yes'] * data_encoded['bmi']
data_encoded['children_age_interaction'] = data_encoded['children'] * data_encoded['age']

# Step 4: Handle outliers (using IQR method)
Q1 = data_encoded[data.select_dtypes(include=['float64', 'int64']).columns].quantile(0.10)
Q3 = data_encoded[data.select_dtypes(include=['float64', 'int64']).columns].quantile(0.90)
IQR = Q3 - Q1

outlier_condition = ((data_encoded[data.select_dtypes(include=['float64', 'int64']).columns] < (Q1 - 1.5 * IQR)) | 
                     (data_encoded[data.select_dtypes(include=['float64', 'int64']).columns] > (Q3 + 1.5 * IQR)))

data_encoded = data_encoded[~outlier_condition.any(axis=1)]

# Step 5: Normalize numerical variables
scaler = MinMaxScaler()

# Include new interaction terms in numerical columns
numerical_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns.drop('charges')  # Exclude 'charges' for separate scaling
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Scale the target variable (charges) separately
y_scaler = MinMaxScaler()
data_encoded['charges'] = y_scaler.fit_transform(data_encoded[['charges']])


# Step 6: Split the data
X = data_encoded.drop(["charges", "sex_male"], axis=1)
y = data_encoded["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)


# PHASE 3: Implement the regression model

def compute_cost(X, y, weights):
    """Compute the cost function with numerical stability."""
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    
    m = len(y)
    predictions = np.dot(X, weights)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error * error)
    return cost



def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Perform gradient descent with weight history tracking for multivariable linear regression.

    Parameters:
    - X: np.ndarray, shape (m, n) -- Feature matrix (m examples, n features)
    - y: np.ndarray, shape (m, 1) -- Target values
    - theta: np.ndarray, shape (n, 1) -- Initial weights (parameters)
    - learning_rate: float -- Learning rate for gradient descent
    - iterations: int -- Number of iterations to run gradient descent

    Returns:
    - theta: np.ndarray, shape (n, 1) -- Optimized weights
    - cost_history: list -- History of the cost function during iterations
    - theta_history: list -- History of weights (parameters) during iterations
    """

    m = len(y)  # Number of training examples
    cost_history = []  # To store cost function values
    theta_history = []  # To store weights (parameters) at each iteration

    for i in range(iterations):
        # Compute predictions
        predictions = np.dot(X, theta)
        
        # Compute the error
        errors = predictions - y
        
        # Compute the gradient
        gradient = (1 / m) * np.dot(X.T, errors)
        
        # Update the parameters
        theta -= learning_rate * gradient
        
        # Store the current weights and cost
        theta_history.append(theta.copy())
        cost = (1 / (2 * m)) * np.sum(errors**2)
        cost_history.append(cost)
        
        if np.isnan(cost):
            print(f"Training stopped at iteration {i} due to numerical instability")
            break

    return theta, cost_history, theta_history

def newton_first_order(X, y, theta, learning_rate, iterations):
    """
    Perform optimization using Newton's first-order method (steepest descent) with weight history tracking.

    Parameters:
    - X: np.ndarray, shape (m, n) -- Feature matrix (m examples, n features)
    - y: np.ndarray, shape (m, 1) -- Target values
    - theta: np.ndarray, shape (n, 1) -- Initial weights (parameters)
    - learning_rate: float -- Learning rate
    - iterations: int -- Number of iterations

    Returns:
    - theta: np.ndarray, shape (n, 1) -- Optimized weights
    - cost_history: list -- History of cost values
    - theta_history: list -- History of weights (parameters) during iterations
    """

    m = len(y)
    cost_history = []
    theta_history = []

    for i in range(iterations):
        # Compute predictions
        predictions = np.dot(X, theta)
        
        # Compute the gradient
        gradient = (1 / m) * np.dot(X.T, (predictions - y))
        
        # Update weights using gradient
        theta -= learning_rate * gradient
        
        # Store weights and cost
        theta_history.append(theta.copy())
        cost = (1 / (2 * m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        if np.isnan(cost):
            print(f"Training stopped at iteration {i} due to numerical instability")
            break

    return theta, cost_history, theta_history

# Newton's Method (Second Order)
def newton_second_order(X, y, theta, iterations):
    """
    Perform optimization using Newton's second-order method with weight history tracking.

    Parameters:
    - X: np.ndarray, shape (m, n) -- Feature matrix (m examples, n features)
    - y: np.ndarray, shape (m, 1) -- Target values
    - theta: np.ndarray, shape (n, 1) -- Initial weights (parameters)
    - iterations: int -- Number of iterations

    Returns:
    - theta: np.ndarray, shape (n, 1) -- Optimized weights
    - cost_history: list -- History of cost values
    - theta_history: list -- History of weights (parameters) during iterations
    """

    m = len(y)
    cost_history = []
    theta_history = []

    for i in range(iterations):
        # Compute predictions
        predictions = np.dot(X, theta)
        
        # Compute the gradient
        gradient = (1 / m) * np.dot(X.T, (predictions - y))
        
        # Compute the Hessian matrix
        hessian = (1 / m) * np.dot(X.T, X)
        
        # Update weights using Newton's step
        theta -= np.linalg.inv(hessian).dot(gradient)
        
        # Store weights and cost
        theta_history.append(theta.copy())
        cost = (1 / (2 * m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        if np.isnan(cost):
            print(f"Training stopped at iteration {i} due to numerical instability")
            break

    return theta, cost_history, theta_history


def plot_contours(X, y, weights_history, title):
    """Modified contour plotting function for high-dimensional data."""
    # Use only first two features for visualization
    X_subset = X[:, :2]
    weights_subset = np.array(weights_history)[:, :2]
    
    w0 = np.linspace(-2, 2, 100)
    w1 = np.linspace(-2, 2, 100)
    W0, W1 = np.meshgrid(w0, w1)
    
    costs = np.zeros_like(W0)
    
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w = np.array([[W0[i, j]], [W1[i, j]]])
            costs[i, j] = compute_cost(X_subset, y, w)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(W0, W1, costs, levels=50, cmap='viridis')
    plt.colorbar(label='Cost')
    
    plt.plot(weights_subset[:, 0], weights_subset[:, 1], 
             marker='o', color='red', label='Update Path')
    
    plt.title(title)
    plt.xlabel('Weight 0')
    plt.ylabel('Weight 1')
    plt.legend()
    plt.show()

# Prepare data for regression
X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Convert to numpy arrays with proper shapes
# X_train_with_bias = np.array(X_train_with_bias, dtype=np.float64)
# X_test_with_bias = np.array(X_test_with_bias, dtype=np.float64)
y_train_array = np.array(y_train.values, dtype=np.float64).reshape(-1, 1)
y_test_array = np.array(y_test.values, dtype=np.float64).reshape(-1, 1)

# Initialize weights and hyperparameters
initial_weights = np.zeros((X_train_with_bias.shape[1], 1), dtype=np.float64)
initial_weights_1 = np.random.randn(X_train_with_bias.shape[1], 1)
initial_weights_2 = np.random.randn(X_train_with_bias.shape[1], 1)  
initial_weights_3 = np.random.randn(X_train_with_bias.shape[1], 1)  
learning_rate = 0.015  # Lowered learning rate
iterations = 1000

# Run all optimization methods
print("\nTraining models...")

print("\nTraining Gradient Descent...")
final_weights_gd, cost_history_gd, weights_history_gd = gradient_descent(
    X_train_with_bias, y_train_array, initial_weights.copy(), learning_rate, iterations
)

print("\nTraining Newton's Method (First Order)...")
final_weights_newton_first, cost_history_newton_first, weights_history_newton_first = newton_first_order(
    X_train_with_bias.copy(), y_train_array.copy(), initial_weights.copy(), learning_rate, iterations
)

print("\nTraining Newton's Method (Second Order)...")
final_weights_newton_second, cost_history_newton_second, weights_history_newton_second = newton_second_order(
    X_train_with_bias.copy(), y_train_array.copy(), initial_weights.copy(), iterations
)

# Convert the weight histories from list to NumPy arrays
weights_history_gd = np.array(weights_history_gd)
weights_history_newton_first = np.array(weights_history_newton_first)
weights_history_newton_second = np.array(weights_history_newton_second)

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot gradient descent weights history
for i in range(weights_history_gd.shape[1]):
    axes[0].plot(weights_history_gd[:, i], label=f'Weight {i+1}')
axes[0].set_title("Gradient Descent Weights History", fontsize=14)
axes[0].set_xlabel("Epochs", fontsize=12)
axes[0].set_ylabel("Weight Value", fontsize=12)
axes[0].grid(True)
axes[0].legend()

# Plot Newton's first order weights history
for i in range(weights_history_newton_first.shape[1]):
    axes[1].plot(weights_history_newton_first[:, i], label=f'Weight {i+1}')
axes[1].set_title("Newton's First Order Weights History", fontsize=14)
axes[1].set_xlabel("Epochs", fontsize=12)
axes[1].set_ylabel("Weight Value", fontsize=12)
axes[1].grid(True)
axes[1].legend()

# Plot Newton's second order weights history
for i in range(weights_history_newton_second.shape[1]):
    axes[2].plot(weights_history_newton_second[:, i], label=f'Weight {i+1}')
axes[2].set_title("Newton's Second Order Weights History", fontsize=14)
axes[2].set_xlabel("Epochs", fontsize=12)
axes[2].set_ylabel("Weight Value", fontsize=12)
axes[2].grid(True)
axes[2].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# Plot contours for each method
plot_contours(X_train_with_bias, y_train_array, weights_history_gd, "Gradient Descent Contour Plot")
plot_contours(X_train_with_bias, y_train_array, weights_history_newton_first, "Newton First Order Contour Plot")
plot_contours(X_train_with_bias, y_train_array, weights_history_newton_second, "Newton Second Order Contour Plot")



# PHASE 4: Model Evaluation

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    """
    Evaluates the model's performance by calculating Mean Squared Error (MSE) and R-squared (R²).
    
    Parameters:
    - y_true: np.ndarray, shape (m,) -- True target values
    - y_pred: np.ndarray, shape (m,) -- Predicted target values
    
    Returns:
    - mse: float -- Mean Squared Error (MSE)
    - r2: float -- R-squared (R²)
    """
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate R-squared (R²)
    r2 = r2_score(y_true, y_pred)
    
    return mse, r2

# Generate predictions for each method
y_pred_gd = X_test_with_bias.dot(final_weights_gd)  # Gradient Descent predictions
y_pred_1st = X_test_with_bias.dot(final_weights_newton_first)  # Newton's First Order predictions
y_pred_2nd = X_test_with_bias.dot(final_weights_newton_second)  # Newton's Second Order predictions

# Calculate MSE for the baseline model (mean of y_test_array)
baseline_predictions = np.full_like(y_test_array, np.mean(y_test_array), dtype=np.float64)
mse_baseline = mean_squared_error(y_test_array, baseline_predictions)
print(f"Baseline Model MSE: {mse_baseline}")

# Now evaluate the models using MSE and R²
mse_gd, r2_gd = evaluate_model(y_test_array, y_pred_gd)
print(f"Gradient Descent - MSE: {mse_gd}, R²: {r2_gd}")

mse_1st, r2_1st = evaluate_model(y_test_array, y_pred_1st)
print(f"Newton's First Order - MSE: {mse_1st}, R²: {r2_1st}")

mse_2nd, r2_2nd = evaluate_model(y_test_array, y_pred_2nd)
print(f"Newton's Second Order - MSE: {mse_2nd}, R²: {r2_2nd}")


# Analyze cost history for all three methods in separate charts within a single frame
plt.figure(figsize=(18, 6))

# Subplot for Gradient Descent
plt.subplot(1, 3, 1)
plt.plot(range(len(cost_history_gd)), cost_history_gd, color='blue')
plt.title("Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

# Subplot for Newton's First Order
plt.subplot(1, 3, 2)
plt.plot(range(len(cost_history_newton_first)), cost_history_newton_first, color='red')
plt.title("Newton's First Order")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

# Subplot for Newton's Second Order
plt.subplot(1, 3, 3)
plt.plot(range(len(cost_history_newton_second)), cost_history_newton_second, color='green')
plt.title("Newton's Second Order")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

# Adjust layout and display
plt.suptitle("Cost History during Optimization (All Methods)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the title
plt.show()

