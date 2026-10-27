import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from CSV file
df = pd.read_csv("1housing.csv")

# Separate features and target
X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['MEDV']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Evaluate the model
print("Actual values:", list(y_test))
print("Predicted values:", y_pred.round(2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Predict for a new house
new_house = pd.DataFrame(
    [[6.5, 5.0, 15.5]],
    columns=['RM', 'LSTAT', 'PTRATIO']
)

print("Predicted MEDV:", round(model.predict(new_house)[0], 2))