import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
df = pd.DataFrame(data, columns=['X', 'Y'])


X = df[['X']]  
y = df['Y']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

coefficient = model.coef_[0]

covariance = np.cov(df['X'], df['Y'])[0][1]


mean_X = X.mean()[0]
mean_Y = y.mean()
variance_X = X.var()[0]  
variance_Y = y.var()    

plt.scatter(X, y, color='blue', label='Data Points') 
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

print(f'Coefficient (Slope): {coefficient}')
print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Covariance between X and Y: {covariance}')
print(f'Mean of X: {mean_X}')
print(f'Mean of Y: {mean_Y}')
print(f'Variance of X: {variance_X}')
print(f'Variance of Y: {variance_Y}')

new=np.array([[2]])
print(new)
pred = model.predict(new)
print(pred)