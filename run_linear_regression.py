#! usr/bin/python
# -*- codding:utf-8 -*-

from linear_regression import linearRegression as lrm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    x, y = make_regression(7000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    linear = lrm(x.shape[1])
    linear.train(x_train, y_train)
    y_pre = linear.predict(x_test)
    print('tensorflow r2:', r2_score(y_pre.ravel(), y_test.ravel()))

    lr = LinearRegression()
    y_pred = lr.fit(x_train, y_train).predict(x_test)
    print('sklearn r2:', r2_score(y_pred.ravel(), y_test.ravel()))



