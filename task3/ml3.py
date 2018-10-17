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
from sklearn.neighbors import KNeighborsClassifier


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def euclidean(a, b):
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i])**2
    return math.sqrt(distance)


def manhattan(a_, b_):
    distance = 0
    for a, b in zip(a_, b_):
        distance += 0 if a == b else 1
    return distance


class KNeighboursClassifier:
    def __init__(self, n_neighbours=1, metric=euclidean):
        self.n_neighbours = n_neighbours
        self.metric = metric

    def fit(self, X_learn, y_learn):
        self.X_learn = X_learn
        self.y_learn = y_learn

    def predict(self, X_predict):
        print('predict:')
        y_predict = []

        for x in X_predict:
            distanses = []
            for x_l, y in zip(X_learn, y_learn):
                distanses.append((self.metric(x, x_l), y))
            sum = 0
            for r in sorted(distanses, key=lambda x: x[0])[:self.n_neighbours]:
                sum += r[1]
            avg = sum / (self.n_neighbours)
            y_predict.append(round(avg))

        return np.array(y_predict)


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
        max = column.max()
    except ValueError:
        print("except ValueError")
        mode = 5.0

    print(f"mode = {mode}")
    print(f"mean = {mean}")
    print(f"max = {max}")
    print("---")

    if mode == -1:
        mode = mean

    res = column.map(lambda x: mode if np.isnan(x) else x)

    #
    # res = res.map(lambda x: x / max)

    return res


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
    # super_feature = pd.get_dummies(df['primaryTitle'])
    # print(super_feature.head())
    # return super_feature

    # f0 = processing_floats(df['numVotes']) # numVotes
    # f1 = processing_floats(df['runtimeMinutes'])

    # f2 = processing_geners(df['genres'])
    f2 = pd.get_dummies(df['genres'])

    # f3 = processing_title_type(df['titleType'])
    f3 = pd.get_dummies(df['titleType'])

    f4 = pd.get_dummies(df['MainWriterPrimaryProfession'])

    f5 = pd.get_dummies(df['firstShownCountry'])
    # f6 = pd.get_dummies(df['prefVotersCountry'])
    # firstShownCountry

    f7 = pd.get_dummies(df['isAdult'])
    f8 = pd.get_dummies(df['worldPromotion'])

    return pd.concat([f2, f3, f4, f5, f7, f8], axis=1)


def accuracy_score(y_true, y_predict):
    """
    :param y_true: Correct target values
    :param y_predict: Estimated target values
    :return: accuracy_score(y_true, y_predict)
    """
    count = 0
    for x, y in zip(y_true, y_predict):
        count += 1 if x == y else 0

    return count / len(y_true)


def learn_test(X, Y):
    """
    Split the set (X, Y) into parts of the same size
    :param X: np.array
    :param Y: np.array
    :return: RMSE(y_test, Y_predict)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    neigh.fit(X_train, y_train)
    Y_predict = neigh.predict(X_test)
    # print([skm.intercept_, *skm.coef_])
    return accuracy_score(y_test, Y_predict)


if __name__ == '__main__':
    df = pd.read_csv('train_knn.csv', sep=',')
    df_predict = pd.read_csv('test_knn.csv', sep=',')

    y = processing_floats(df['rating'])

    df_big = df_pandas_processing(pd.concat([df, df_predict], axis=0, sort=False)) # axis=0, sort=False

    df = df_big[:2000]
    df_predict = df_big[2000:]

    min_max_scaler = preprocessing.MinMaxScaler()
    #array = df.values
    #X = min_max_scaler.fit_transform(array[:, :-1])

    print(df_big.shape)
    print(df.shape)
    print(df_predict.shape)
    print()

    df = pd.concat([df, y], axis=1)

    # getting np.array:
    array = df.values

    X_learn = array[:, :-1] # min_max_scaler.fit_transform(array[:, :-1])
    y_learn = array[:, -1]

    # (1) fit:
    neigh = KNeighborsClassifier(n_neighbors=600, metric='manhattan')
    neigh.fit(X_learn, y_learn)
    # kNN = KNeighboursClassifier(n_neighbours=1, metric=manhattan)
    # kNN.fit(X_learn, y_learn)

    # print(np.shape(X_learn))

    # (2) predict:
    X_predict = df_predict.values

    y_predict = neigh.predict(X_learn)

    print(X_learn.shape)
    print(X_predict.shape)

    # y_predict = neigh.predict(np.concatenate((X_predict, X_learn), axis=0))
    y_predict = neigh.predict(X_predict)

    res_df = pd.DataFrame(y_predict)
    print(res_df.head())
    res_df.to_csv("foo_k600.csv", sep=',')

    # CV:
    n = 100
    for k in [600]:
        neigh = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        sum = 0

        for i in range(n):
            sum += learn_test(X_learn, y_learn)
        print(f'k = {k}')
        print(f'avg = {sum / n}')

