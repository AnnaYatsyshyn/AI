import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії:")
print("Коефіцієнти: ", regr.coef_)
print("Перехоплення: ", regr.intercept_)

# Розрахунок та виведення R^2 (коефіцієнт детермінації)
r2 = r2_score(ytest, ypred)
print("\nR^2 (коефіцієнт детермінації): {:.2f}".format(r2))

# Розрахунок та виведення середньої абсолютної похибки (MAE)
mae = mean_absolute_error(ytest, ypred)
print("Середня абсолютна похибка (MAE): {:.2f}".format(mae))

# Розрахунок та виведення середньоквадратичної похибки (MSE)
mse = mean_squared_error(ytest, ypred)
print("Середньоквадратична похибка (MSE): {:.2f}".format(mse))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
