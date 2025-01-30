import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Linear Regression Model GUI")
st.sidebar.header("Options")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())


    # Data Analysis Section
    if st.sidebar.checkbox("Data Analysis"):
        st.subheader("Dataset Information")
        st.write(data.info())
        
        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        st.write(missing_values)

        st.subheader("Basic Statistics")
        st.write(data.describe())

        st.subheader("Boxplots of Numerical Features")
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            st.write(f"Boxplot for {col}")
            fig, ax = plt.subplots()
            sns.boxplot(y=data[col], ax=ax)
            st.pyplot(fig)  # Explicitly passing the figure
        
        st.subheader("Histograms of Numerical Features")
        fig, ax = plt.subplots(figsize=(12, 8))
        data.select_dtypes(include=['float64', 'int64']).hist(bins=20, figsize=(12, 8), edgecolor='black', ax=ax)
        plt.suptitle("Histograms of Numerical Features", fontsize=16)
        st.pyplot(fig)  # Explicitly passing the figure

        st.subheader("Statistical Summary for Numerical Columns")
        stats_summary = {}
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            stats_summary[col] = {
                "Mean": data[col].mean(),
                "Median": data[col].median(),
                "Variance": data[col].var(),
                "Standard Deviation": data[col].std()
            }

        for col, stats in stats_summary.items():
            st.write(f"**Column: {col}**")
            for stat_name, stat_value in stats.items():
                st.write(f"{stat_name}: {stat_value}")

        st.subheader("Pairplot for Selected Features")
        pairplot_fig = sns.pairplot(data, vars=["age", "bmi", "children", "charges"], hue="smoker", palette="coolwarm")
        st.pyplot(pairplot_fig.fig)  # Explicitly passing the pairplot's figure

        st.subheader("Correlation Matrix")
        correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Correlation Matrix of Numerical Features", fontsize=16)
        st.pyplot(fig)  # Explicitly passing the figure

        st.subheader("BMI Distribution")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data['bmi'], kde=True, bins=30, ax=ax)
        plt.title("Distribution of BMI", fontsize=16)
        st.pyplot(fig)  # Explicitly passing the figure


    # Preprocess Data Section
    if st.sidebar.checkbox("Preprocess the Data"):
        st.subheader("Step 1: Handle Missing Values")
        st.write("Dropping rows with missing values...")
        data_cleaned = data.dropna()
        st.write("Shape of cleaned data:", data_cleaned.shape)
        
        st.subheader("Step 2: Encode Categorical Variables")
        st.write("Performing one-hot encoding for categorical columns: sex, smoker, region")
        data_encoded = pd.get_dummies(data_cleaned, columns=["sex", "smoker", "region"], drop_first=True)
        data_encoded = data_encoded.astype(float)
        st.write("Sample of encoded data:")
        st.write(data_encoded.head())

        st.subheader("Step 3: Add New Features")
        st.write("Creating interaction terms...")
        data_encoded['age_bmi_interaction'] = data_encoded['age'] * data_encoded['bmi']
        data_encoded['smoker_bmi_interaction'] = data_encoded['smoker_yes'] * data_encoded['bmi']
        data_encoded['children_age_interaction'] = data_encoded['children'] * data_encoded['age']
        st.write("Sample with new features added:")
        st.write(data_encoded[['age', 'bmi', 'smoker_yes', 'age_bmi_interaction', 'smoker_bmi_interaction', 'children_age_interaction']].head())

        st.subheader("Step 4: Handle Outliers")
        st.write("Using the IQR method to detect and remove outliers...")
        Q1 = data_encoded[data.select_dtypes(include=['float64', 'int64']).columns].quantile(0.10)
        Q3 = data_encoded[data.select_dtypes(include=['float64', 'int64']).columns].quantile(0.90)
        IQR = Q3 - Q1

        outlier_condition = ((data_encoded[data.select_dtypes(include=['float64', 'int64']).columns] < (Q1 - 1.5 * IQR)) | 
                     (data_encoded[data.select_dtypes(include=['float64', 'int64']).columns] > (Q3 + 1.5 * IQR)))

        data_encoded = data_encoded[~outlier_condition.any(axis=1)]
        st.write("Shape after removing outliers:", data_encoded.shape)

        st.subheader("Step 5: Normalize Numerical Variables")
        st.write("Normalizing numerical columns using MinMaxScaler...")
        scaler = MinMaxScaler()

        # Include new interaction terms in numerical columns
        numerical_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
        numerical_columns = numerical_columns.drop('charges')  # Exclude 'charges' for separate scaling
        data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

        # Scale the target variable (charges) separately
        y_scaler = MinMaxScaler()
        data_encoded['charges'] = y_scaler.fit_transform(data_encoded[['charges']])
        st.write("Sample data after normalization:")
        st.write(data_encoded[['age', 'bmi', 'charges']].head())

        st.subheader("Step 6: Split the Data")
        st.write("Splitting the data into training and testing sets...")
        X = data_encoded.drop(["charges", "sex_male"], axis=1)
        y = data_encoded["charges"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"Training set shape: {X_train.shape}")
        st.write(f"Test set shape: {X_test.shape}")

        st.subheader("Visualize the Preprocessed Data")
        if st.checkbox("Show Correlation Heatmap"):
            correlation_matrix = data_encoded.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        if st.checkbox("Show Histogram of Charges"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.histplot(data_encoded['charges'], kde=True, bins=30, ax=ax)
            plt.title("Distribution of Charges", fontsize=16)
            st.pyplot(fig)


    # Train the models with selected parameters
        if st.sidebar.checkbox("Train Models"):
            def compute_cost(X, y, weights):
                X = np.array(X, dtype=np.float64)
                y = np.array(y, dtype=np.float64)
                weights = np.array(weights, dtype=np.float64)
                m = len(y)
                predictions = np.dot(X, weights)
                error = predictions - y
                cost = (1/(2*m)) * np.sum(error * error)
                return cost

            def gradient_descent(X, y, theta, learning_rate, iterations):
                m = len(y)
                cost_history = []
                theta_history = []

                for i in range(iterations):
                    predictions = np.dot(X, theta)
                    errors = predictions - y
                    gradient = (1 / m) * np.dot(X.T, errors)
                    theta -= learning_rate * gradient
                    theta_history.append(theta.copy())
                    cost = (1 / (2 * m)) * np.sum(errors**2)
                    cost_history.append(cost)
                    
                    if np.isnan(cost):
                        print(f"Training stopped at iteration {i} due to numerical instability")
                        break

                return theta, cost_history, theta_history

            def newton_first_order(X, y, theta, learning_rate, iterations):
                m = len(y)
                cost_history = []
                theta_history = []

                for i in range(iterations):
                    predictions = np.dot(X, theta)
                    gradient = (1 / m) * np.dot(X.T, (predictions - y))
                    theta -= learning_rate * gradient
                    theta_history.append(theta.copy())
                    cost = (1 / (2 * m)) * np.sum((predictions - y)**2)
                    cost_history.append(cost)
                    
                    if np.isnan(cost):
                        print(f"Training stopped at iteration {i} due to numerical instability")
                        break

                return theta, cost_history, theta_history

            def newton_second_order(X, y, theta, iterations):

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

                    #added because of the error
                    hessian += np.eye(hessian.shape[0]) * 1e-5  # Add a small value to the diagonal

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
                """Generate contour plots for regression models."""
                X_subset = X[:, :2]  # Use only the first two features for visualization
                weights_subset = np.array(weights_history)[:, :2]  # Use only the first two weights

                w0 = np.linspace(-2, 2, 100)
                w1 = np.linspace(-2, 2, 100)
                W0, W1 = np.meshgrid(w0, w1)

                costs = np.zeros_like(W0)

                for i in range(W0.shape[0]):
                    for j in range(W0.shape[1]):
                        w = np.array([[W0[i, j]], [W1[i, j]]])
                        costs[i, j] = compute_cost(X_subset, y, w)

                fig, ax = plt.subplots(figsize=(10, 6))
                contour = ax.contourf(W0, W1, costs, levels=50, cmap='viridis')
                plt.colorbar(contour, label='Cost')
                ax.plot(weights_subset[:, 0], weights_subset[:, 1], marker='o', color='red', label='Update Path')
                ax.set_title(title)
                ax.set_xlabel('Weight 0')
                ax.set_ylabel('Weight 1')
                ax.legend()
                st.pyplot(fig)


            # Streamlit GUI Section
            st.title("Regression Model Training with Gradient Descent and Newton's Method")

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

            # Sliders for parameters
            learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=learning_rate, step=0.001)
            iterations = st.slider("Iterations", min_value=100, max_value=5000, value=iterations, step=100)

            final_weights_gd, cost_history_gd, weights_history_gd = gradient_descent(
                X_train_with_bias, y_train_array, initial_weights.copy(), learning_rate, iterations
            )

            final_weights_newton_first, cost_history_newton_first, weights_history_newton_first = newton_first_order(
                X_train_with_bias.copy(), y_train_array.copy(), initial_weights.copy(), learning_rate, iterations
            )
            
            final_weights_newton_second, cost_history_newton_second, weights_history_newton_second = newton_second_order(
                X_train_with_bias.copy(), y_train_array.copy(), initial_weights.copy(), iterations
            )

            # Convert the weight histories from list to NumPy arrays
            weights_history_gd = np.array(weights_history_gd)
            weights_history_newton_first = np.array(weights_history_newton_first)
            weights_history_newton_second = np.array(weights_history_newton_second)

            # Plotting Results
            st.subheader("Cost Function History")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cost_history_gd, label="Gradient Descent")
            ax.plot(cost_history_newton_first, label="Newton's First Order")
            ax.plot(cost_history_newton_second, label="Newton's Second Order")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Cost")
            ax.set_title("Cost Function History")
            ax.legend()
            st.pyplot(fig)

            # Plotting Weight History
            st.subheader("Weight History")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Plot Gradient Descent weight history
            for i in range(weights_history_gd.shape[1]):
                axes[0].plot(weights_history_gd[:, i], label=f'Weight {i+1}')
            axes[0].set_title("Gradient Descent Weights History")
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel("Weight Value")
            axes[0].grid(True)
            axes[0].legend()

            # Plot Newton's First Order weight history
            for i in range(weights_history_newton_first.shape[1]):
                axes[1].plot(weights_history_newton_first[:, i], label=f'Weight {i+1}')
            axes[1].set_title("Newton's First Order Weights History")
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("Weight Value")
            axes[1].grid(True)
            axes[1].legend()

            # Plot Newton's Second Order weight history
            for i in range(weights_history_newton_second.shape[1]):
                axes[2].plot(weights_history_newton_second[:, i], label=f'Weight {i+1}')
            axes[2].set_title("Newton's Second Order Weights History")
            axes[2].set_xlabel("Epochs")
            axes[2].set_ylabel("Weight Value")
            axes[2].grid(True)
            axes[2].legend()

            st.pyplot(fig)

            # Contour plots for each method
            st.write("### Contour Plots")
            plot_contours(X_train_with_bias, y_train_array, weights_history_gd, "Gradient Descent Contour Plot")
            plot_contours(X_train_with_bias, y_train_array, weights_history_newton_first, "Newton First Order Contour Plot")
            plot_contours(X_train_with_bias, y_train_array, weights_history_newton_second, "Newton Second Order Contour Plot")



            # Model Evaluation Section
            if st.sidebar.checkbox("Evaluate Model"):
                st.subheader("Model Evaluation")

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
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    return mse, r2

                # Prepare test data
                X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
                y_test_array = y_test.values.reshape(-1, 1)

                # Evaluate Gradient Descent
                y_pred_gd = np.dot(X_test_with_bias, final_weights_gd)
                mse_gd, r2_gd = evaluate_model(y_test_array, y_pred_gd)
                st.write("### Gradient Descent")
                st.write(f"Mean Squared Error (MSE): {mse_gd}")
                st.write(f"R-squared (R²): {r2_gd}")

                # Evaluate Newton's First Order
                y_pred_1st = np.dot(X_test_with_bias, final_weights_newton_first)
                mse_1st, r2_1st = evaluate_model(y_test_array, y_pred_1st)
                st.write("### Newton's First Order")
                st.write(f"Mean Squared Error (MSE): {mse_1st}")
                st.write(f"R-squared (R²): {r2_1st}")

                # Evaluate Newton's Second Order
                y_pred_2nd = np.dot(X_test_with_bias, final_weights_newton_second)
                mse_2nd, r2_2nd = evaluate_model(y_test_array, y_pred_2nd)
                st.write("### Newton's Second Order")
                st.write(f"Mean Squared Error (MSE): {mse_2nd}")
                st.write(f"R-squared (R²): {r2_2nd}")
