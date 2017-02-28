import sklearn
import pandas
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model

from openpyxl import load_workbook
wb = load_workbook(filename = 'Hector.xlsx')
sheet_ranges = wb['TIPO 1']
print wb.get_sheet_names()
#print(sheet_ranges['D18'].value)

df = pandas.DataFrame(wb['TIPO 1'].values, columns=['x_1','x_2','y'])
df2=df[2:]
print df2
X=np.array(df2[['x_1','x_2']])
y=np.array(df2[['y']])
experiments=10
for i in range(experiments):

	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	# Fit regression model
	rng = np.random.RandomState(1)
	regr_1 = DecisionTreeRegressor(max_depth=4)

	regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
	regr_3 = linear_model.Ridge (alpha = .5)
	regr_4 = linear_model.Lasso(alpha = 0.1)

	regr_1.fit(X_train, y_train)
	regr_2.fit(X_train, y_train)
	regr_3.fit(X_train, y_train)
	regr_4.fit(X_train, y_train)


	# Predict
	y_1 = regr_1.predict(X_test)
	y_2 = regr_2.predict(X_test)
	y_3 = regr_3.predict(X_test)
	y_4 = regr_4.predict(X_test)

	# Plot the results
	plt.figure()
	x_axis=np.array(range(1,len(y_test)+1))
	plt.scatter(x_axis, y_test, c="k", label="training samples")
	plt.plot(x_axis, y_1, c="g", label="DTR n=1", linewidth=2)
	plt.plot(x_axis, y_2, c="r", label="AB-DTR n=300", linewidth=2)
	plt.plot(x_axis, y_3, c="b", label="ridge", linewidth=2)
	plt.plot(x_axis, y_3, c="y", label="LASSO", linewidth=2)
	plt.xlabel("data")
	plt.ylabel("target")
	plt.title("Boosted Decision Tree Regression")
	plt.legend()
	plt.show()
