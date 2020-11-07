# regression
linear regression
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
import sklearn



df = pd.read_csv("http://bit.ly/w-data")
df.head()


df.describe()

sns.regplot(x='Hours',y='Scores',data=df)
plt.title('Hours vs Percentage',fontsize=20)  
plt.xlabel('Hours Studied',fontsize=10)  
plt.ylabel('Percentage Score',fontsize=10)   
plt.show()


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

##linear regression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression  
clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

y_pred

df1= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df1

hours = np.array([9.25])
hours = hours.reshape(-1,1)
own_pred = clf.predict(hours)
print(f"No of Hours = {hours}")
print(f"Predicted Score = {own_pred[0]}")

##error metrics

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)
print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
