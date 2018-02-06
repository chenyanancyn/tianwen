import csv  # 读取csv文件时用到
import numpy as np
from normalization_self import normalization_self
 

def train_index_numpy(use='unnormal'):
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
        if count == 50:
            break

    train_index_1 = np.array(train_index_0)
    train_index = np.reshape(train_index_1, (483851, 2))
    # train_index = np.reshape(train_index_1, (50, 2))
    # print(train_index)


    # 读取txt文件，并按照train_index矩阵的名字进行排序
    path_ini = '../first_train_data_20180131/'
    Efield_0 = []

    # count = 0
    for i in range(train_index.shape[0]):
        # if train_index[i][0] == 744889:
        #     continue
        # else:
        path_mid = str(train_index[i][0])
        path = path_ini + path_mid + '.txt'
        with open(path, 'r') as fr:
            lines = fr.readline()  # 整行读取数据
            pos = [float(i) for i in lines.split(',')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            # pos.insert(0, train_index[i][1])  # 进行归一化后，再进行拼接
            Efield_0.append(pos)

                # count = count + 1
                # if count == 50:
                #     break
        
    Efield = np.array(Efield_0)
    # print(Efield.shape)
    # print(train_index[:, 1].shape)
    if use == 'normalization':
        return Efield
    else:
        # 对矩阵进行归一化处理
        Efield = normalization_self(Efield)
        # 对归一化后的矩阵进行拼接label
        label = np.reshape(train_index[:, 1], (50, 1))
        Efield = np.concatenate((label, Efield), axis=1)  # axis=1表示对应行的数组进行拼接
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

        return train[0:48385], train[48385:96770], train[96770:145155], train[145155:193540], train[193540:241925], train[241925:290310], train[290310:338695], train[338695:387080], validation, test
        # return train[0:10], train[10:20], train[20:30], train[30:40], train[40:50], train[50:60], train[60:70], train[70:80], validation, test


train_index_numpy(use='unnormal')