import csv  # 读取csv文件时用到
import numpy as np
 
def test_numpy():
    # 读取csv文件，并存为numpy矩阵
    csv_reader = csv.reader(open('../first_test_index_20180131.csv', encoding='utf-8'))
    test_index_0 = []
    for row in csv_reader:
        if row != ['id']:
            test_index_0.append(int(row[0]))

    test_index_1 = np.array(test_index_0)
    test_index = np.reshape(test_index_1, (100000, 1))

    # 读取txt文件，并按照test_index矩阵的名字进行排序
    path_ini = '../first_test_data_20180131/'
    Efield_0 = []

    # count = 0
    for i in range(test_index.shape[0]):
        path_mid = str(test_index[i][0])
        path = path_ini + path_mid + '.txt'
        with open(path, 'r') as fr:
            lines = fr.readline()  # 整行读取数据
            pos = [float(i) for i in lines.split(',')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            Efield_0.append(pos)
        # count = count + 1
        # if count == 1000:
        #     break
    Efield = np.array(Efield_0)
    # print(Efield.shape)

    return Efield[0:50000], Efield[50000:]
    # return Efield[0:500], Efield[500:]
