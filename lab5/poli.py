import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Загрузка csv файла
data = pd.read_csv('dataset.csv')

# Определение зависимой переменной (атрибута, который нужно предсказать)
target_variable = 'pts'

# Определение независимых переменных (атрибутов, от которых зависит целевой атрибут)
independent_variables = ['age', 'player_height', 'player_weight']

# Разделение данных на обучающую и тестовую выборки
X = data[independent_variables]
y = data[target_variable]
# Разделение данных на обучающий набор и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =================== PART 2 ======================

# Преобразование независимых переменных в полиномиальные признаки
degree = 2  # Степень полинома
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_poly, y)

# Получаем предсказания модели для тестовых данных
y_pred = model.predict(X_test_poly)

# Вычисляем R^2 score для модели
score = r2_score(y_test, y_pred)

# Выводим значение score
print("R² Score:", score)

plt.scatter(X['player_height'], y, label='Исходные данные')
plt.plot(X['player_height'], model.predict(X_poly), color='blue', label='Полиномиальная регрессия')
plt.xlabel('player_height')
plt.ylabel(target_variable)
plt.title('Полиномиальная регрессия')
plt.legend()
plt.show()

