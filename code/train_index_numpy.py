import csv  # 读取csv文件时用到
import numpy as np
 

def train_index_numpy():
    # 读取csv文件，并存为numpy矩阵
    dict_0 = {'galaxy': 0, 'star': 1, 'qso': 2, 'unknown': 3}
    csv_reader = csv.reader(open('../first_train_index_20180131.csv', encoding='utf-8'))
    train_index_0 = []
    count = 0
    for row in csv_reader:
        if row[0] == 'id':
            continue
        else:
            temp = []
            temp.append(int(row[0]))
            temp.append(dict_0[row[1]])
            train_index_0.append(temp)
        count = count + 1
        if count == 1000:
            break

    train_index_1 = np.array(train_index_0)
    # train_index = np.reshape(train_index_1, (483851, 2))
    train_index = np.reshape(train_index_1, (1000, 2))
    # print(train_index)


    # 读取txt文件，并按照train_index矩阵的名字进行排序
    path_ini = '../first_train_data_20180131/'
    Efield_0 = []

    count = 0
    for i in range(train_index.shape[0]):
        if train_index[i][0] == 744889:
            continue
        else:
            path_mid = str(train_index[i][0])
            path = path_ini + path_mid + '.txt'
            with open(path, 'r') as fr:
                lines = fr.readline()  # 整行读取数据
                pos = [float(i) for i in lines.split(',')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                pos.insert(0, train_index[i][1])
                Efield_0.append(pos)
            print(count)
        count = count + 1
        if count == 1000:
            break
    Efield = np.array(Efield_0)
    print(Efield.shape)


    # 分割训练集，训练集：验证集：测试集 = 8：1：1
    train_0 = []
    validation_0 = []
    test_0 = []
    for i in range(Efield.shape[0]):
        if i % 10 == 3:
            validation_0.append(Efield[i])
        elif i % 10 == 8:
            test_0.append(Efield[i])
        else:
            train_0.append(Efield[i])

    # 列表转换为numpy
    train = np.array(train_0)
    validation = np.array(validation_0)
    test = np.array(test_0)
    # print(train.shape)
    # print(validation.shape)
    # print(test.shape)

    # return train[0:48385], train[48385:96770], train[96770:145155], train[145155:193540], train[193540:241925], train[241925:290310], train[290310:338695], train[338695:387080], validation, test
    return train[0:100], train[100:200], train[200:300], train[300:400], train[400:500], train[500:600], train[600:700], train[700:800], validation, test
