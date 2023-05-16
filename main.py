import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dataset.csv')
target_variable = 'pts'
independent_variables = ['age', 'player_height', 'player_weight']

X = data[independent_variables]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Средняя квадратичная ошибка (MSE):', mse)
print('Коэффициент детерминации (R^2):', r2)

sns.scatterplot(x='age', y=target_variable, data=data)
plt.plot(X['age'], model.predict(X), color='blue')
plt.xlabel('age')
plt.ylabel(target_variable)
plt.title('Линейная регрессия')
plt.show()


