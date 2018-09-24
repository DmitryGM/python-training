import csv
import numpy as np

from pip.utils import encoding


def csv_reader(file_obj):
    """
    Read a csv file
    """
    data = []

    reader = csv.reader(file_obj, delimiter=';')
    for row in reader:
        data.append(row)
    return data[1:]


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


csv_path = "students.csv"
with open(csv_path, encoding='utf8') as f_obj:
    data = csv_reader(f_obj)

res = processing(data)

for row in res:
    print('\t'.join(row))


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
