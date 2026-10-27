import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
df = pd.read_csv("adult.csv")

# Separate features and target
X = df.drop("income", axis=1)
y = df["income"]

# Convert categorical data into numeric data
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Decision Tree model
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Predict income
y_pred = model.predict(X_test)

# Evaluate model
print("Actual values:", list(y_test))
print("Predicted values:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Confusion Matrix:\n", confusion_matrix(
    y_test, y_pred, labels=model.classes_
))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, zero_division=0
))

plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)
plt.show()