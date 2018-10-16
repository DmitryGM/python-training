import copy
import math
import csv
import random
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.stats as stats


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def try_parse_to_float(string):
    """
    :return: float(string)
    """
    try:
        x = float(str(string).replace(',', '.'))
        return x
    except ValueError:
        return float('nan')


def processing_floats(column):
    """
    :return: Column filled floats
    """
    column = column.map(try_parse_to_float)
    try:
        mode = float((column.mode()[0]))
        mean = column.mean()
    except ValueError:
        print("except ValueError")
        mode = 5.0

    print("mode =", mode)
    print("mean =", mean)
    print("---")

    if mode == -1:
        mode = mean

    return column.map(lambda x: mode if np.isnan(x) else x)


def processing_geners(column):
    """ genre classification """
    d = {'Family': [],
         'Drama': [],
         'Horror': [],
         'Mystery': [],
         'Documentary': [],
         'Comedy': []
         }

    for string in column:
        for x in d:
            if x in string:
                d[x].append(1)
            else:
                d[x].append(0)

    return pd.DataFrame(data=d)


def processing_title_type(column):
    """ movie type clustering """
    s = set()
    for string in column:
        s |= {string}
    print('set_title_type:')
    print(s)

    d = dict([(elem, []) for elem in s])
    print(d)

    for string in column:
        for x in d:
            if x == string:
                d[x].append(1)
            else:
                d[x].append(0)

    df = pd.DataFrame(data=d)
    print(df)
    return df


def processing_votes(num_votes, gender_voters):
    a = list(map(lambda x: x[1]/x[0], list(zip(num_votes, gender_voters))))
    print(a)
    return pd.Series(a)


def df_pandas_processing(df):
    """
    Selecting features from data
    :param df: DataFrame -- our data set
    :return: DataFrame containing only selected features
    """

    # features:
    f1 = processing_floats(df['runtimeMinutes'])
    f2 = processing_geners(df['genres'])
    f3 = processing_title_type(df['titleType'])

    return pd.concat([f1, f2, f3], axis=1)


def rmse(y_true, y_predict):
    """
    :param y_true: Correct target values
    :param y_predict: Estimated target values
    :return: RMSE(y_true, y_predict)
    """
    return math.sqrt(mean_squared_error(y_true, y_predict))


def learn_test(X, Y):
    """
    Split the set (X, Y) into parts of the same size
    :param X: np.array
    :param Y: np.array
    :return: RMSE(y_test, Y_predict)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    skm.fit(X_train, y_train)
    Y_predict = skm.predict(X_test)
    # print([skm.intercept_, *skm.coef_])
    return rmse(y_test, Y_predict)


def get_coef(X, Y):
    """
    Split the set (X, Y) into parts of the same size
    :param X: np.array
    :param Y: np.array
    :return: coefs
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    skm.fit(X_train, y_train)

    return ([skm.intercept_, *skm.coef_])


df_predict = pd.read_csv('test.csv', sep=',')

df = pd.read_csv('train.csv', sep=',')
y = processing_floats(df['averageRating'])
df = df_pandas_processing(df)
df = pd.concat([df, y], axis=1)

# getting np.array:
array = df.values

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(array[:, :-1])
Y = array[:, -1]


# (1) FILL:
# scikit-learn:
print('scikit-learn:')
skm = lm.LinearRegression()
skm.fit(X, Y)
print([skm.intercept_, *skm.coef_])


# (2) predict:
Y_predict = skm.predict(X)
print('Y_predict')
print(Y_predict)

# (2)*:
def predict(df):
    X = min_max_scaler.fit_transform(df.values[:, :])
    res = skm.predict(X)

    res_df = pd.DataFrame(res)
    res_df.to_csv("foo.csv", sep=',')

predict(df_pandas_processing(df_predict))


# (3) RMSE:
print('RMSE:')
print(rmse(Y, Y_predict))


# (4) learn/test:
print(learn_test(X, Y))


# (5) CV:
sum = 0
n = 10000

for i in range(n):
    sum += learn_test(X, Y)
print(sum / n)
