import csv
import time
import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

X = [] # Dates
y = [] # Delta Days
annotations = [] # Patches

def getData(filename):
    with open(filename, 'r') as csvfile:
        csvFilereader = csv.reader(csvfile)

        # Skip Header
        next(csvFilereader)
        for row in csvFilereader:
            y.append(float(re.sub(r'\.', '', row[2])));
            X.append(time.mktime(datetime.datetime.strptime(row[1], "%d.%m.%Y").timetuple()) / 10E6)
            annotations.append(str(row[0]))
    return

def predictRelease(dates, timeDelta):

    # Real Values
    dates = np.reshape(dates, (len(dates), 1))
    maxDate = max(dates) * 10E6

    # Finding likely next release date
    step = (sum(timeDelta) / len(timeDelta)) / sum(timeDelta)
    step = (1 - step) * step
    averageDelta = 0

    while averageDelta <= 1:
        averageDelta += step

    dates = preprocessing.MinMaxScaler().fit_transform(np.array(dates))

    # Plot Background
    plt.rcParams['axes.facecolor'] = '#424143'

    # Calculating predictions
    svr_lin = SVR(kernel= 'linear', C=1E-13)
    svr_lin.fit(dates, timeDelta)

    svr_poy = SVR(kernel= 'poly', C=1E2, degree = 4)
    svr_poy.fit(dates, timeDelta)

    svr_rbf = SVR(kernel='rbf', C=1E3, gamma = 1E10)
    svr_rbf.fit(dates, timeDelta)

    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(dates, timeDelta)

    svr_lin_k = svr_lin.predict(averageDelta)
    svr_poy_k = svr_poy.predict(averageDelta)
    svr_rbf_k = svr_rbf.predict(averageDelta)
    lin_reg_k = linear_reg.predict(averageDelta)

    pred_step = (1 / max([svr_lin_k, svr_poy_k, svr_rbf_k, lin_reg_k])) * (averageDelta - 1)

    # Ploting & Stuff
    plt.plot(dates, svr_lin.predict(dates), color='#BEDB39', label = 'Linear')

    #--- Linear
    plt.scatter(1 + pred_step * svr_lin_k, svr_lin_k, color='#BEDB39', label = 'LinearP')
    plt.annotate(
        datetime.datetime.fromtimestamp((svr_lin_k * 8.64E4 + maxDate)).strftime('%d-%m-%Y'),
        xy = (1 + pred_step * svr_lin_k, svr_lin_k)
    )
    print("Linear Prediction: %.0f" % svr_lin_k)

    #--- Polynomial
    plt.plot(dates, svr_poy.predict(dates), color='#FF5347', label = 'Polynomial')
    #Prediction
    plt.scatter(1 + pred_step * svr_poy_k, svr_poy_k, color='#FF5347', label = 'PolynomialP')
    plt.annotate(
        datetime.datetime.fromtimestamp((svr_poy_k * 8.64E4 + maxDate)).strftime('%d-%m-%Y'),
        xy = (1 + pred_step * svr_poy_k, svr_poy_k)
    )
    #Output
    print("Polynomial Prediction: %.0f " % svr_poy_k)

    #--- RBF
    plt.plot(dates, svr_rbf.predict(dates), color='#1F8A70', label = 'RBF')
    # Prediction
    plt.scatter(1 + pred_step * svr_rbf_k, svr_rbf_k, color='#1F8A70', label = 'RBFP')
    plt.annotate(
        datetime.datetime.fromtimestamp((svr_rbf_k * 8.64E4 + maxDate)).strftime('%d-%m-%Y'),
        xy = (1 + pred_step * svr_rbf_k, svr_rbf_k)
    )
    # Output
    print("RBF Prediction: %.0f " % svr_rbf_k)

    #--- Linear Regression

    plt.plot(dates, linear_reg.predict(dates), color='cyan', label = 'LR')
    # Prediction
    #--averageDelta
    plt.scatter(1 + pred_step * lin_reg_k, lin_reg_k, color='cyan', label = 'LRP')
    plt.annotate(
        datetime.datetime.fromtimestamp((lin_reg_k * 8.64E4 + maxDate)).strftime('%d-%m-%Y'),
        xy = (1 + pred_step * lin_reg_k, lin_reg_k)
    )
    # Output
    print("Linear Regression Prediction: %.0f " % lin_reg_k)

    # Real Data
    plt.scatter(dates, timeDelta, color='white', label = 'Real Releases')
    for i in range(len(dates)):
        plt.annotate(
            annotations[i],
            xy = (dates[i], timeDelta[i])
        )

    # Legend
    plt.xlabel('Patch Release Timestamp (Normalized)')
    plt.ylabel('Days')
    plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    plt.show()

    return

#Run
getData('patches.csv')
predictRelease(X, y)
