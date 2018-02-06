import numpy as np


def normalization_self(matrix_0):
    # 从mean_std文件中读出均值和标准差
    with open('mean_std.txt', 'r') as fr:
        mean_s = fr.readline().strip('\n')
        mean_l = [float(i) for i in mean_s.split(' ')]
        mean = np.array(mean_l)

        std_s = fr.readline().strip('\n')
        std_l = [float(i) for i in std_s.split(' ')]
        std = np.array(std_l)
    # 带入公式，并进行归一化处理
    shape_0 = matrix_0.shape
    matrix = np.zeros(shape=(shape_0[0], shape_0[1]))
    for i in range(matrix_0.shape[0]):
        matrix[i] = (matrix_0[i] - mean)/std

    return matrix


# a = np.array([[1, 2, 3], [4, 5, 6]])
# matrix = normalization(a)
# print(type(matrix))
# print(matrix)
