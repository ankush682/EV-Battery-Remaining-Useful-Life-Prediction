
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("/content/Battery_RUL.csv")

# Display basic info
print(df.shape)
print(df.head())
print(df.tail())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Basic statistics
print("\nSummary statistics:")
print(df.describe())


#removing the last row cuz it is the sum of all the above rows
df.drop(index=15063,inplace=True)

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#boxplot to check outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1]):  # Skip the 'RUL' column
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(column, fontsize=10)
    plt.tight_layout()

plt.suptitle("Boxplots for Outlier Detection (Features Only)", fontsize=16, y=1.03)
plt.show()

# handling the outliers
df_capped = df.copy()
outlier_columns = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)"
]

# IQR capping function
def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply IQR capping to selected columns
for col in outlier_columns:
    df_capped[col] = cap_outliers_iqr(df_capped[col])

# Display capped values summary for confirmation
df_capped[outlier_columns].describe()

# boxplot to check the after result of the iqr
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_capped.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df_capped[column], color='skyblue')
    plt.title(column, fontsize=10)
    plt.tight_layout()

plt.suptitle("Boxplots for Outlier Detection (Features Only)", fontsize=16, y=1.03)
plt.show()

# Scaling the data and splitting the data into x and y

X = df_capped.drop("RUL", axis=1)
y = df_capped["RUL"]

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to a DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Show a few rows of the scaled data
X_scaled_df.head()

# X = your scaled features, y = target (RUL)
# training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,       # features
    y,              # target
    test_size=0.2,  # 20% test data
    random_state=42 # reproducible results
)

# Confirm the shapes
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# hyperparameter for ridge


ridge = Ridge()

# Define parameter grid
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

# Grid Search with 5-fold cross-validation
ridge_grid = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
ridge_grid.fit(X_train, y_train)

# Best model
best_ridge = ridge_grid.best_estimator_
print("✅ Best alpha for Ridge:", ridge_grid.best_params_)

# Predict and evaluate
y_pred_ridge = best_ridge.predict(X_test)

# applying the ridge model


ridge_model = Ridge(alpha=0.1)

# 2. Train the model
ridge_model.fit(X_train, y_train)

# 3. Predict on test set
y_pred_ridge = ridge_model.predict(X_test)

# 4. Evaluate performance
mse = mean_squared_error(y_test, y_pred_ridge)
mae = mean_absolute_error(y_test, y_pred_ridge)
r2 = r2_score(y_test, y_pred_ridge)

print("📊 Ridge Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# check for overfitting!!!!!!!!!!!!!!!!!!!!!!!!!


# Train R²
y_train_pred = ridge_model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

# Test R² (already computed)
r2_test = r2_score(y_test, y_pred_ridge)

print(f"R² Train Score: {r2_train:.4f}")
print(f"R² Test Score : {r2_test:.4f}")

# hyperparameter tuning for lasso

lasso = Lasso(max_iter=10000)

# Parameter grid
param_grid_lasso = {
    'alpha': [0.001, 0.01, 0.1, 1, 10]
}

# Grid search
lasso_grid = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='r2')
lasso_grid.fit(X_train, y_train)

# Best model
best_lasso = lasso_grid.best_estimator_
print("✅ Best alpha for Lasso:", lasso_grid.best_params_)

# Predict and evaluate
y_pred_lasso = best_lasso.predict(X_test)

# applying the lasso model



lasso_model = Lasso(alpha=0.001)

# 2. Fit the model
lasso_model.fit(X_train, y_train)

# 3. Predict on test set
y_pred_lasso = lasso_model.predict(X_test)

# 4. Evaluate

mse = mean_squared_error(y_test, y_pred_lasso)
mae = mean_absolute_error(y_test, y_pred_lasso)
r2 = r2_score(y_test, y_pred_lasso)

print("📊 Lasso Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# hyperparameter tuning for decision tree


tree = DecisionTreeRegressor(random_state=42)

# Define parameter grid
param_grid_tree = {
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
tree_grid = GridSearchCV(tree, param_grid_tree, cv=5, scoring='r2')
tree_grid.fit(X_train, y_train)

# Best model
best_tree = tree_grid.best_estimator_
print("✅ Best parameters for Decision Tree:", tree_grid.best_params_)

# Predict and evaluate
y_pred_tree = best_tree.predict(X_test)

# applying the decision tree model

# 1. Initialize model (set max_depth to prevent overfitting)
tree_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=5, random_state=42)

# 2. Train the model
tree_model.fit(X_train, y_train)

# 3. Predict on test set
y_pred_tree = tree_model.predict(X_test)

# 4. Evaluate
mse = mean_squared_error(y_test, y_pred_tree)
mae = mean_absolute_error(y_test, y_pred_tree)
r2 = r2_score(y_test, y_pred_tree)

print("📊 Decision Tree Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# hyperparameter tuning for random forest

# Define the model
rf = RandomForestRegressor(random_state=42)

# Define parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid Search
rf_grid = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Best model
best_rf = rf_grid.best_estimator_
print("✅ Best parameters for Random Forest:", rf_grid.best_params_)

# Predict and evaluate
y_pred_rf = best_rf.predict(X_test)

#applying the random forest model

# 1. Initialize the model
rf_model = RandomForestRegressor(max_depth=15, min_samples_leaf=1, min_samples_split=2, n_estimators=150, random_state=42)

# 2. Train the model
rf_model.fit(X_train, y_train)

# 3. Predict on test set
y_pred_rf = rf_model.predict(X_test)

# 4. Evaluate
mse = mean_squared_error(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("📊 Random Forest Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# applying the svr model



# 1. Initialize SVR model
svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)

# 2. Train the model
svr_model.fit(X_train, y_train)

# 3. Predict on test set
y_pred_svr = svr_model.predict(X_test)

# 4. Evaluate
mse = mean_squared_error(y_test, y_pred_svr)
mae = mean_absolute_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)

print("📊 SVR Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# XGBoost Regression Model


#  Create and train the model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # regression objective
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

#  Predict on test data
y_pred_xgb = xgb_model.predict(X_test)

#  Evaluate the model
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)

print("📊 XGBoost Evaluation Metrics:")
print("R² Score:", r2_score(y_test, y_pred_xgb))
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("RMSE:", rmse)

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.title("Feature Importance - XGBoost")
plt.show()

# applying the knn model


# 1. Initialize model
knn_model = KNeighborsRegressor(n_neighbors=5)

# 2. Fit the model
knn_model.fit(X_train, y_train)

# 3. Predict
y_pred_knn = knn_model.predict(X_test)

# 4. Evaluate
mse = mean_squared_error(y_test, y_pred_knn)
mae = mean_absolute_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)

print("📊 KNN Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

#the graph for the random forest



plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.4, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("📈 Actual vs Predicted RUL (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()

#applying the linear regression model

# Train linear model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score
r2_linear = r2_score(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("📊 Linear Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Predictions from all models (already trained and predicted earlier)
model_names = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "SVR (RBF Kernel)",
    "KNN Regressor"
]

predictions = [
    y_pred_lr,
    y_pred_ridge,
    y_pred_lasso,
    y_pred_tree,
    y_pred_rf,
    y_pred_svr,
    y_pred_knn
]

mse_values = [mean_squared_error(y_test, pred) for pred in predictions]
mae_values = [mean_absolute_error(y_test, pred) for pred in predictions]
r2_scores = [r2_score(y_test, pred) for pred in predictions]

comparison_df = pd.DataFrame({
    "Model": model_names,
    "MSE": mse_values,
    "MAE": mae_values,
    "R² Score": r2_scores
}).sort_values(by="MSE", ascending=True).reset_index(drop=True)

comparison_df

