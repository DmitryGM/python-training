import math
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy as pt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pip.utils import encoding

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


def csv_pandas_reader(filename):
    """
    Parse a csv file
    """
    df = pd.read_csv(filename, sep=';')
    # print(df.head(0))
    # print(df.info())
    # print(df.describe())
    # print(df['пол'].value_counts())
    # print(df.mean())
    # print("!!!")
    return df


def processing_city(column):
    return column.map(lambda x: 1 if x == 'A' else 0)


def processing_months(column):
    # Родился в мае?
    return column.map(lambda x: 1 if 5 == x else 0)


def processing_floats(column):
    column = column.map(try_parse_to_float)
    try:
        mode = float((column.mode()[0]))
    except ValueError:
        mode = 5.0

    return column.map(lambda x: mode if x == -1 else x)


def processing_two_numbers(column):
    pass


def processing_beer_classifications(column):
    return column.map(lambda x: 0 if str(x) == '0' else 1)


def df_pandas_processing(df):
    # print("df_pandas_processing:")

    # df['город рождения'] = df['город рождения'].apply(lambda x: 1 if x == 'A' else 0)
    # df['город учебы в школе'] = processing_city(df['город учебы в школе'])
    # df['месяц рождения'] = processing_months(df['месяц рождения'])

    # df['средний школьный балл(математика)'] = processing_floats(df['средний школьный балл(математика)'])

    # df['дорога до вуза(время)']
    # df['потребление пива'] = processing_beer_classifications(df['потребление пива'])

    # print(df.shape)
    # print(df.info())

    f1 = processing_floats(df['оценка по математике (школа)'])
    f2 = processing_floats(df['средний школьный балл(математика)'])
    f3 = processing_floats(df['рост'])
    res = processing_floats(df['Y(оценка по математике в первом семестре)'])
    return pd.concat([f1, f2, f3, res], axis=1)


def rmse(y_true, y_predict):
    """
    :param y_true: Correct target values
    :param y_predict: Estimated target values
    :return: RMSE(y_true, y_predict)
    """
    return math.sqrt(mean_squared_error(y_true, y_predict))


def learn_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    skm.fit(X_train, y_train)
    Y_predict = skm.predict(X_test)

    return rmse(y_test, Y_predict)


df = csv_pandas_reader("students.csv")
df = df_pandas_processing(df)
print(df)

a = df.values
print(a)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(a[:, :-1])
Y = a[:, -1]
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

# statsmodels:
print('statsmodels:')
x_ = sm.add_constant(X)
smm = sm.OLS(Y, x_)
res = smm.fit()
print(res.params)


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
for i in range(1000):
    sum += learn_test(X, Y)
print(sum / 1000)
