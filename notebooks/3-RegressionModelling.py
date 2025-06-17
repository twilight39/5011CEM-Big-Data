# In this notebook, we attempt to create a regression model using the preprocessed mortality data from Malaysia. The model will predict the mortality count based on various features such as year, age group, and disease category.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load the processed data
model_data = pd.read_csv("../data/processed/malaysia_mortality_data_model.csv")

# Define the features (X) and target (y) for regression
# The target is 'Mortality Count'
y = model_data["Mortality Count"]

# The features are everything else. We need to one-hot encode the disease category.
X = model_data.drop("Mortality Count", axis=1)
X = pd.get_dummies(
    X, columns=["Disease_L2"], prefix="Disease"
)  # Turn diseases into features

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
# 3. Initialize and train a regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_regressor.fit(X_train, y_train)

# %%
# 4. Evaluate the regression model
y_pred_rf = rf_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest Regressor R-squared: {r2:.4f}")
print(f"Random Forest Regressor Mean Absolute Error: {mae:.2f}")

# The Random Forest Regressor achieved an R-squared value of 0.8819, indicating that our model can explain 88.2% of the variability in mortality counts—a very strong result.

# The Mean Absolute Error (MAE) is 181.37 deaths. While this number seems high in isolation, its impact is context-dependent.
# For major causes of death like Cardiovascular Disease, where counts are in the tens of thousands, an error of ~181 is negligible and allows for highly accurate resource planning.
# For rarer diseases, this error margin is larger, which is an expected trade-off. Overall, the model demonstrates high predictive power where it is most critically needed.

# %%
# Let's check the feature importances to see which diseases are most important for predicting mortality!

# Check feature importances
feature_importances = pd.Series(
    rf_regressor.feature_importances_, index=X.columns
).sort_values(ascending=False)

# Print top 10
print("\nFeature Importances from Random Forest Regressor:")
print(feature_importances.head(10))

# Unsurprisingly, 'Age Group' is the most important feature, followed by 'Cardiovascular diseases'.
# Surprisingly, 'Year' is also a significant feature, indicating that mortality trends have indeed change oved time.
