import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR

data = pd.read_csv('dataset.csv')

target_variable = 'pts'

independent_variables = ['age', 'reb', 'ast']

X = data[independent_variables]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =================== PART 3 ======================

model = SVR(kernel='linear')
model.fit(X_train, y_train)
svr_linear_score = model.score(X_test, y_test)
print("SVR (Linear Kernel) Score:", svr_linear_score)

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_train, y_train)
svr_rbf_score = svr_rbf.score(X_test, y_test)
print("SVR (RBF Kernel) Score:", svr_rbf_score)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = []

for kernel in kernels:
    model = SVR(kernel=kernel)
    model.fit(X_scaled, y)
    models.append(model)

plt.figure(figsize=(12, 8))

for i, model in enumerate(models):
    plt.subplot(2, 2, i+1)
    plt.scatter(X['reb'], y, color='green', label='Исходные данные')
    plt.plot(X['reb'], model.predict(X_scaled), color='blue', label=f'SVM ({kernels[i]})')
    plt.xlabel('reb')
    plt.ylabel(target_variable)
    plt.title(f'SVM с ядром {kernels[i]}')
    plt.legend()

plt.tight_layout()
plt.show()