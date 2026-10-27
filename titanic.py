import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# Convert categorical columns into numeric columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)

# Handle missing values
df = df.dropna()

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict survival
y_pred = model.predict(X_test)

# Evaluate model
print("Actual values:", list(y_test))
print("Predicted values:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Plot confusion matrix as a heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Did not survive', 'Survived'],
    yticklabels=['Did not survive', 'Survived']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Titanic Survival Confusion Matrix')
plt.show()

# Predict for a new passenger
new_passenger = pd.DataFrame([{
    'Pclass': 3,
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.05,
    'Sex_male': 1,
    'Embarked_Q': 0,
    'Embarked_S': 1
}], columns=X.columns)

prediction = model.predict(new_passenger)[0]

print("Survival Prediction:", "Survived" if prediction == 1 else "Did not survive")