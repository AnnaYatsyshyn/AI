import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)


def plot_learning_curves(model, x, Y, title):
    X_train, X_val, y_train, y_val = train_test_split(x, Y, test_size=0.2)
    train_errors, val_errors = [], []
    for M in range(1, len(X_train)):
        model.fit(X_train[:M], y_train[:M])
        y_train_predict = model.predict(X_train[:M])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:M]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r+-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.title(title)
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, title="Linear Regression Learning Curves")
plt.show()

poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg, X, y, title="Polynomial Regression (Degree=10) Learning Curves")
plt.show()

poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg_2, X, y, title="Polynomial Regression (Degree=2) Learning Curves")
plt.show()
