from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# Load dataset
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
svm_reg = SVR(kernel='rbf')
svm_reg.fit(X_train, y_train)

# Predict & Evaluate
y_pred = svm_reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
