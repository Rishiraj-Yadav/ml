import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict income
y_pred = model.predict(X_test)

# Evaluate model
print("Actual values:", list(y_test))
print("Predicted values:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted Income")
plt.ylabel("Actual Income")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# Visualize one tree from the Random Forest
plt.figure(figsize=(16, 8))
plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Random Forest - One Decision Tree")
plt.tight_layout()
plt.show()