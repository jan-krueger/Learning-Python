import csv
import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
import re

dates = []
prices = []

def getData(filename):
    with open(filename, 'r') as csvfile:
        csvFilereader = csv.reader(csvfile)
        next(csvFilereader)
        x = 0;
        for row in csvFilereader:
            try:
                prices.append(float(re.sub(r'\$|,|[[:space:]]', '', row[1])));
                dates.append(x)
                x += 1
            except ValueError:
                print("ValueError at %d" % (x))
                break
    return

def predictPrices(dates, prices, x):

    # Real Values
    dates = np.reshape(dates, (len(dates), 1))

    dates = preprocessing.MinMaxScaler().fit_transform(np.array(dates).reshape(-1, 1))
    prices = preprocessing.MinMaxScaler().fit_transform(np.array(prices).reshape(-1, 1))

    # Ploting & Stuff
    print("Doing linear...")
    svr_lin = SVR(kernel= 'linear', C=1E4)
    svr_lin.fit(dates, prices)
    plt.plot(dates, svr_lin.predict(dates), color='#468966', label = 'Linear')
    print("Linear done...")

    print("Doing polynomial...")
    svr_poy = SVR(kernel= 'poly', C=1E4, degree = 3)
    svr_poy.fit(dates, prices)
    plt.plot(dates, svr_poy.predict(dates), color='#FFF0A5', label = 'Polynomial')
    print("Polynomial done...")

    print("Doing RBF...")
    svr_rbf = SVR(kernel='rbf', C=1E8, gamma = 1E-1)
    svr_rbf.fit(dates, prices)
    plt.plot(dates, svr_rbf.predict(dates), color='#FFB03B', label = 'RBF')
    print("RBF done...")

    print("Doing Sigmoid...")
    svr_sig = SVR(kernel='sigmoid', C=1E12, gamma = 1E-1)
    svr_sig.fit(dates, prices)
    plt.plot(dates, svr_sig.predict(dates), color='#B64926', label = 'Sigmoid')
    print("Sigmoid done...")

    print("Doing Linear Regression...")
    bodyReg = linear_model.LinearRegression()
    bodyReg.fit(dates, prices)

    plt.plot(dates, bodyReg.predict(dates), color='cyan', label = 'Linear Regression')
    print("Linear Regression Done...")

    plt.scatter(dates, prices, color='black', label = 'Data')

    plt.xlabel('Date')
    plt.ylabel('Money')
    plt.legend()
    plt.show()

    result = [];
    result.append(['Linear', svr_lin.predict(x)[0]]);
    result.append(['Linear Regression', bodyReg.predict(x)[0]])
    result.append(['Polynomia', svr_poy.predict(x)[0]]);
    result.append(['RBF', svr_rbf.predict(x)[0]]);
    return result;

getData('data-new.csv')
predicted_price = predictPrices(dates, prices, 2000)
print(predicted_price)
