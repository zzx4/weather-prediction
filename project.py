import numpy as np
from matplotlib import pyplot as plt
import pandas

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

#alphas = [3, 1, 0.3, 0.1, 0.03, 0.01]
alphas = [0.01, 0.03, 0.1, 0.3, 1, 3] #better for plotting

data = pandas.read_csv("weather_data.csv")

data = data.drop(['Row ID', 'Station ID', 'Station Name', 'Station Province', 'Date'], axis=1)

scaler = StandardScaler()
to_scale = ['Year', 'Month', 'Day']
data[to_scale] = scaler.fit_transform(data[to_scale])

X = data.drop('Mean Temperature (C)', axis=1)
y = data['Mean Temperature (C)']
y_filtered = y.dropna() #get rid of NaNs
X_filtered = X.loc[y_filtered.index] #remove dropped entries

#print(data.head())
#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=2)


#baseline
y_mean = np.mean(y)
baseline_pred = np.ones_like(y_test)
baseline_pred = baseline_pred * y_mean
base_rmse = root_mean_squared_error(y_test, baseline_pred)
print(base_rmse)


#stochastic gradient descent
train_error = []
for a in alphas:
    #sgd = SGDRegressor(max_iter=1000, alpha=0.03, random_state=2)
    sgd = SGDRegressor(max_iter=1000, alpha=a, random_state=2)  #specify random state for reproducible results
    sgd.fit(X_train, y_train)

    y_pred = sgd.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    train_error.append(rmse)
    #mae = mean_absolute_error(y_test, y_pred2)
    #print(mse2, mae)
    #print(rmse, a)
    #print(y_test.std())

#sgd plot
plt.plot(alphas, train_error, linestyle='dashed', marker='.')
plt.xlabel("alpha")
plt.ylabel("root mean squared error")
plt.show()

#random forest regression
trees = [100, 200, 500, 800]
forest_rmses = []
for t in trees:
    forest = RandomForestRegressor(random_state=2, n_estimators=t)
    forest.fit(X_train, y_train)
    y_forest = forest.predict(X_test)

    forestrmse = root_mean_squared_error(y_test, y_forest)
    forest_rmses.append(forestrmse)

    #print(forestrmse)

#forest plot
plt.plot(trees, forest_rmses, linestyle='dashed', marker='.')
plt.xlabel("n_estimators")
plt.ylabel("root mean squared error")
plt.show()

#mlp
mlp_alphas = [0.00001, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
mlp_rmses = []
for a in mlp_alphas:
    mlp = MLPRegressor(activation='logistic', random_state=2, alpha=a, max_iter=1000) #logistic provided the best result
    mlp.fit(X_train, y_train)

    y_mlp = mlp.predict(X_test)

    mlprmse = root_mean_squared_error(y_test, y_mlp)
    mlp_rmses.append(mlprmse)

    print(mlprmse, a)

#mlp plot
plt.plot(mlp_alphas, mlp_rmses, linestyle='dashed', marker='.')
plt.xlabel("alpha")
plt.ylabel("root mean squared error")
plt.show()