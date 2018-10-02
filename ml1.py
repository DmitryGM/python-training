import math
import csv
import random
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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
        return -1


def processing_city(column):
    """
    :return: 1 if city is St. Petersburg, else -- 0
    """
    return column.map(lambda x: 1 if x == 'A' else 0)


def processing_months(column):
    """
    :return: Column filled 0 and 1
    """
    #return column.map(lambda x: 1 if 6 <= x <= 8 else 0)
    return column.map(lambda x: random.randint(0,100))


def processing_floats(column):
    """
    :return: Column filled floats
    """
    column = column.map(try_parse_to_float)
    try:
        mode = float((column.mode()[0]))
    except ValueError:
        mode = 5.0

    return column.map(lambda x: mode if x == -1 else x)


def processing_beer_classifications(column):
    """
    :return: Column filled 0 and 1
    """
    return column.map(lambda x: 0 if str(x) == '0' else 1)


def df_pandas_processing(df):
    """
    Selecting features from data
    :param df: DataFrame -- our data set
    :return: DataFrame containing only selected features
    """
    # print("df_pandas_processing:")
    # df['город рождения'] = df['город рождения'].apply(lambda x: 1 if x == 'A' else 0)
    # df['город учебы в школе'] = processing_city(df['город учебы в школе'])
    # df['месяц рождения'] = processing_months(df['месяц рождения'])
    # df['средний школьный балл(математика)'] = processing_floats(df['средний школьный балл(математика)'])
    # df['дорога до вуза(время)']
    # df['потребление пива'] = processing_beer_classifications(df['потребление пива'])
    # print(df.shape)
    # print(df.info())



    # features:
    f1 = processing_floats(df['оценка по математике (школа)'])
    #f2 = processing_floats(df['средний школьный балл(математика)'])
    #f3 = processing_floats(df['рост'])
    #f4 = processing_months(df['месяц рождения'])
    y = processing_floats(df['Y(оценка по математике в первом семестре)'])
    return pd.concat([f1, y], axis=1)


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


df = pd.read_csv("students.csv", sep=';')
df = df_pandas_processing(df)

# getting np.array:
array = df.values

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
