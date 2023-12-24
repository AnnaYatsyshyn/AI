import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X**2 + X + 4 + np.random.randn(m, 1)

plt.scatter(X, y, label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Модель лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Передбачення для побудови лінійної регресії
X_range = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred_lin = lin_reg.predict(X_range)

# Дані та модель лінійної регресії на графіку
plt.scatter(X, y, label='Дані')
plt.plot(X_range, y_pred_lin, color='red', label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Модель поліноміальної регресії
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Передбачення для побудови поліноміальної регресії
X_range_poly = poly_features.transform(X_range)
y_pred_poly = poly_reg.predict(X_range_poly)

# Дані та модель поліноміальної регресії на графіку
plt.scatter(X, y, label='Дані')
plt.plot(X_range, y_pred_poly, color='green', label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Оцінка якості поліноміальної регресії
y_pred_poly_train = poly_reg.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly_train)
mae_poly = mean_absolute_error(y, y_pred_poly_train)
mse_poly = mean_squared_error(y, y_pred_poly_train)

print("R^2 (поліноміальна регресія): {:.2f}".format(r2_poly))
print("Середня абсолютна похибка (поліноміальна регресія): {:.2f}".format(mae_poly))
print("Середньоквадратична похибка (поліноміальна регресія): {:.2f}".format(mse_poly))

# Коефіцієнти полінома
print("Коефіцієнти полінома (X^2, X):", poly_reg.coef_)

# Перехоплення
print("Перехоплення:", poly_reg.intercept_)

# Значення коефіцієнтів
theta0 = poly_reg.intercept_[0]
theta1 = poly_reg.coef_[0][0]
theta2 = poly_reg.coef_[0][1]
print(f"Модель поліноміальної регресії: y = {theta0:.2f} + {theta1:.2f} * X + {theta2:.2f} * X^2")