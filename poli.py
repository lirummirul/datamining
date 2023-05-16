import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')
target_variable = 'pts'
independent_variables = ['age', 'player_height', 'player_weight']

X = data[independent_variables]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =================== PART 2 ======================

degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_test_poly)

score = r2_score(y_test, y_pred)
print("R² Score:", score)

plt.scatter(X['player_height'], y, label='Исходные данные')
plt.plot(X['player_height'], model.predict(X_poly), color='blue', label='Полиномиальная регрессия')
plt.xlabel('player_height')
plt.ylabel(target_variable)
plt.title('Полиномиальная регрессия')
plt.legend()
plt.show()

