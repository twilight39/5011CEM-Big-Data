# In this notebook, we attempt to create a classification model using the preprocessed mortality data from Malaysia. The model will predict disease categories based on various features such as year, age group.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the processed data
model_data = pd.read_csv("../data/processed/malaysia_mortality_data_model.csv")


# Create meaningful target: Mortality Risk Levels
def create_risk_levels(mortality_count: int):
    if mortality_count < 500:
        return "Low"
    elif mortality_count < 2000:
        return "Medium"
    else:
        return "High"


model_data["Risk_Level"] = model_data["Mortality Count"].apply(create_risk_levels)

# %%
# Define the features and target variable for the classification model.

# Features: Age Group + Disease + Year + Sex
# Target: Risk Level
X = pd.get_dummies(
    model_data[["Year", "Age Group", "Disease_L2", "Sex_Females", "Sex_Males"]]
)
y = model_data["Risk_Level"]

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
# Initialize and trains a Decision Tree Classifier on the training data.
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate the model on the test set
print(f"Model Accuracy: {dt_model.score(X_test, y_test)}")

# %%
# The Decision Tree has an accuracy of 90.1%!

# Let's try a more complex model: Random Forest Classifification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"Decision Tree Accuracy: {dt_model.score(X_test, y_test):.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
# %%
# Random Forest Classifier performed slightly better at 92.6% accuracy,
# Let's check the classification report to see how well it performs across different risk levels.
from sklearn.metrics import classification_report

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
# The report reveals a nuanced story. While the model is nearly perfect
# at identifying 'Low' risk cases (99% recall), its ability to correctly
# identify all 'Medium' (62% recall) and 'High' (71% recall) cases is lower.
#
# %%
# To visualize exactly where the model is making mistakes, let's use a confusion matrix!
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)

disp.plot()
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()

# The matrix confirms that the primary source of error is the model confusing 'Medium' and 'High' risk cases with each other.
# For instance, 14 'Medium' risk cases were incorrectly labeled as 'High'.

# %%
# We finally have a working classification model.
# Let's check the feature importances to see what the model thinks is important!

# Create a pandas Series to store feature importances
feature_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("Feature Importances from Random Forest:")
print(feature_importances)

# So it seems the model thinks 'Age Group' and 'Year' are the most important features,
# taking up 33.8% and 8.8% of the importance respectively.
# Following which, Cardiovascular diseases contribute 5.6%.
