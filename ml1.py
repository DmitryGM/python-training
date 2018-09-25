import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pip.utils import encoding

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def try_parse_to_float(string):
    '''
    :return: float(string)
    '''
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




# print("X =")
# print(X)
# Xinv = np.linalg.inv(X)
#
# Xdot = np.dot(X, Xinv)

#Xt = np.transpose(X)


# print(X)
# #print(Xt)
# print(Xinv)
# print(Xdot)
