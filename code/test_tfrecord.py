import tensorflow as tf
# from PIL import Image
# import numpy
# import  matplotlib.pyplot as plt
from test_numpy import test_numpy



# 读取数据
print('loading data...')
data = test_numpy()


# 转为tfrecord文件
config = ['test_z_0', 'test_z_1']
for each in range(len(config)):
    mnist_type = config[each]
    # tfrecord格式文件名
    with tf.python_io.TFRecordWriter('data_' + mnist_type + '.tfrecords') as writer:
        data_path = data[each]
        for num_data in range(data_path.shape[0]):
            temp_data = data[each][num_data]
            print(temp_data)
            # img = img.astype('uint8')
            # image = Image.fromarray(img.reshape(32, 48))
            # fig = plt.figure()
            # plotwindow = fig.add_subplot(111)
            # plt.imshow(image, cmap='gray')    # cmap  图像为灰度图
            # label = data[each][num_image][0]
            # print(label)
            # plt.show()
            data_byte = temp_data.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={   
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_byte]))  
                }))
            writer.write(example.SerializeToString())
print('successful')

