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
import pylab as pl
import scipy.stats as stats


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def draw_hist(list):
    list = sorted(list)
    fit = stats.norm.pdf(list, np.mean(list), np.std(list))  # this is a fitting indeed
    pl.plot(list, fit, '-o')
    pl.hist(list, density=True)
    pl.show()


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
    # s = set()
    # for lst in column:
    #     print(lst)
    #     s |= set(list(lst))
    #     print(set(lst))
    # print(s)

    # 'Short': [],

    d = {'Family': [],
        'Drama': [], 'Horror': [],
         'Mystery': [], 'Documentary': [],
         'Comedy': []
         }

    for string in column:
        for x in d:
            if x in string:
                d[x].append(1)
            else:
                d[x].append(0)

    df = pd.DataFrame(data=d)
    #print(df)
    return df


def processing_title_type(column):
    # s = set()
    # for string in column:
    #     s |= {string}
    # print('set_title_type:')
    # print(s)
    #
    # d = dict([(elem, []) for elem in s])
    # print(d)

    d = {'Drama': [], 'Drama': [],
         }

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

    print("df_pandas_processing")
    # features:

    # best feature:
    f1 = processing_floats(df['runtimeMinutes'])

    f2 = processing_geners(df['genres'])

    # f = processing_title_type(df['titleType'])

    # f = processing_votes(df['numVotes'], df['genderVoters'])
    # f = processing_floats(df['titleLength']) # bad feature
    # f = processing_floats(df['Downloads'])
    # f = processing_floats(df['worldPromotion'])

    y = processing_floats(df['averageRating'])
    return pd.concat([f2, f1, y], axis=1)


def df_pandas_processing_for(df):
    """
    Selecting features from data
    :param df: DataFrame -- our data set
    :return: DataFrame containing only selected features
    """


    # #'Family': [],
    #     'Drama': [], 'Horror': [],
    #      'Mystery': [], 'Documentary': [],
    #      'Comedy': []

    d_short = {'runtimeMinutes': [],
               #'Family': []}
               #'Drama': [],
               'Horror': []}
               #'Mystery': [],
               #'Documentary': [],
               #'Comedy': []}
    d_not_short = copy.deepcopy(d_short)


    target = processing_floats(df['averageRating'])

    for i in range(len(df)):
        # best feature:
        runtimeMinutes = try_parse_to_float(df['runtimeMinutes'][i])

        if df['titleType'][i] == 'short':
            d_not_short['runtimeMinutes'].append(0)
            d_not_short['Horror'].append(0)

            d_short['runtimeMinutes'].append(runtimeMinutes)

            if 'Horror' in df['genres'][i]:
                d_short['Horror'].append(1)
            else:
                d_short['Horror'].append(0)
        else:
            d_short['runtimeMinutes'].append(0)
            d_short['Horror'].append(0)

            d_not_short['runtimeMinutes'].append(runtimeMinutes)

            if 'Horror' in df['genres'][i]:
                d_not_short['Horror'].append(1)
            else:
                d_not_short['Horror'].append(0)


    f_genres = processing_geners(df['genres'])

    df_short = pd.DataFrame(data=d_short)
    df_not_short = pd.DataFrame(data=d_not_short)

    df_short['runtimeMinutes'] = processing_floats(df_short['runtimeMinutes'])
    df_not_short['runtimeMinutes'] = processing_floats(df_not_short['runtimeMinutes'])


    # f2 = processing_geners(df['genres'])

    # f = processing_title_type(df['titleType'])

    # f = processing_votes(df['numVotes'], df['genderVoters'])
    # f = processing_floats(df['titleLength']) # bad feature
    # f = processing_floats(df['Downloads'])
    # f = processing_floats(df['worldPromotion'])

    return pd.concat([df_short, df_not_short, target], axis=1)


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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75)
    skm.fit(X_train, y_train)

    return ([skm.intercept_, *skm.coef_])


df_predict = pd.read_csv('test.csv', sep=',')

df = pd.read_csv('train.csv', sep=',')
df = df_pandas_processing(df)


# getting np.array:
array = df.values

print(array[:5])

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(array[:, :-1])
Y = array[:, -1]
print("X =")
print(X)
print("Y =")
print(Y)


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

Y_predict = skm.predict(X)

# (3) RMSE:
print('RMSE:')
print(rmse(Y, Y_predict))


# (4) learn/test:
print(learn_test(X, Y))


# (5) CV:
sum = 0
n = 1000

for i in range(n):
    sum += learn_test(X, Y)
print(sum / n)


# ---
print("---")

stat = []

for i in range(n):
    stat.append(get_coef(X, Y)[2])

draw_hist(stat)
