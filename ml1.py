import csv
import numpy as np
import pandas as pd

from pip.utils import encoding


def csv_pandas_reader(filename):
    """
    Parse a csv file
    """
    df = pd.read_csv(filename, sep=';')
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    print(df.head(0))
    print(df.info())
    print("!!!")
    return df


def norm(lst):
    lst = list(map(lambda x: float(str(x).replace(",", ".")), lst))
    maximum = max(lst)
    minimum = min(lst)
    # print("max = {0}".format(maximum))
    # print("min = {0}".format(minimum))
    # print()

    return list(map(lambda x: round((x - minimum) / (maximum - minimum), 1), lst))


def processing(data):
    res = np.array(data)

    height = np.array(norm(res[:, 3]))
    average_score = np.array(norm(res[:, 7]))
    math_score = np.array(norm(res[:, 10]))

    res[:, 3] = height
    res[:, 7] = average_score
    res[:, 10] = math_score

    res2 = np.concatenate(height[:, np.newaxis])
    print(res2)

    #print(np.concatenate((height, height, height), axis=0))
    #year_birth = norm(res[:, 4])
    #month_birth = norm(res[:, 5])
    #res[:, 14] = norm(res[:, 14]) # дорога до вуза

    #16 -- ряд в аудитории

    #res[:, 17] = norm(res[:, 17]) # доля пропусков

    # 18 автоматы на экз

    for row in res:
        if row[1] == "A":
            row[1] = 1
        else:
            row[1] = 0

        if row[2] == "A":
            row[2] = 1
        else:
            row[2] = 0

    rows = np.array(list(range(50))) # !!!
    columns = [3, 7, 10]

    return res[rows[:, np.newaxis], columns]


df = csv_pandas_reader("students.csv")

res = processing(df)

X = np.array([[1, 4, 11],
              [5, 2, 23],
              [12, 11, 3]])
Xinv = np.linalg.inv(X)

Xdot = np.dot(X, Xinv)

#Xt = np.transpose(X)


# print(X)
# #print(Xt)
# print(Xinv)
# print(Xdot)
