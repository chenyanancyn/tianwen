import numpy as np
from train_index_numpy import train_index_numpy

# 读取样本矩阵
matrix = train_index_numpy(use='normalization')
# 求取每个特征的均值
mean_0 = matrix.mean(axis=0)
# print(mean_0.shape)
# 将每一个特征的均值字符化
mean_s = mean_0.astype(str)
# 求取每个特征的标准差
std_0 = matrix.std(axis=0)
# print(std_0.shape)
# 将每一个特征的均值字符化
std_s = std_0.astype(str)
# 将得到的均值和标准差写入文件中， 需要将其转换为字符串
f = open('mean_std.txt', 'w')    # r只读，w可写，a追加
mean = ' '.join(mean_s)
f.write(mean+'\n')
std = ' '.join(std_s)
f.write(std+'\n')
f.close()
# # 读出文件中的内容
# with open('mean_std.txt', 'r') as fr:
#     mean_s = fr.readline().strip('\n')
#     mean_l = [float(i) for i in mean_s.split(' ')]
#     mean = np.array(mean_l)

#     std_s = fr.readline().strip('\n')
#     std_l = [float(i) for i in std_s.split(' ')]
#     std = np.array(std_l)

