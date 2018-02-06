import tensorflow as tf
# from PIL import Image
# import numpy
# import  matplotlib.pyplot as plt
from train_index_numpy import train_index_numpy



# 读取数据
print('loading data...')
data = train_index_numpy()


# 转为tfrecord文件
config = ['train_1', 'train_2', 'train_3',  'train_4', 'train_5', 'train_6', 'train_7','train_8', 'validation', 'test']
for each in range(len(config)):
    mnist_type = config[each]
    # tfrecord格式文件名
    with tf.python_io.TFRecordWriter('data_' + mnist_type + '.tfrecords') as writer:
        data_path = data[each]
        for num_data in range(data_path.shape[0]):
            temp_data = data[each][num_data][1:]
            data_byte = temp_data.tobytes()
            label = data[each][num_data][0]
            # print(label)
            example = tf.train.Example(features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),  
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_byte]))  
                }))
            writer.write(example.SerializeToString())
print('successful')