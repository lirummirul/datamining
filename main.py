import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR

# Загрузка csv файла
data = pd.read_csv('dataset.csv')

# Определение зависимой переменной (атрибута, который нужно предсказать)
target_variable = 'pts'

# Определение независимых переменных (атрибутов, от которых зависит целевой атрибут)
independent_variables = ['age', 'player_height', 'player_weight']

# Разделение данных на обучающую и тестовую выборки
X = data[independent_variables]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
# line_reg_score = r2_score(X_test, y_train)
# print('Что-то, что у Али: ', line_reg_score)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка производительности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Средняя квадратичная ошибка (MSE):', mse)
print('Коэффициент детерминации (R^2):', r2)

sns.scatterplot(x='age', y=target_variable, data=data)  # Диаграмма рассеяния
plt.plot(X['age'], model.predict(X), color='blue')  # Линия регрессии
plt.xlabel('age')
plt.ylabel(target_variable)
plt.title('Линейная регрессия')
plt.show()


