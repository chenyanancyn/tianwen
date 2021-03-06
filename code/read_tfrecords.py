# import  matplotlib.pyplot as plt
import tensorflow as tf
# from PIL import Image
import numpy as np

def read_tfrecord(config_dir, num=1):   
    # 读取tfrecord代码      
    filename_queue = tf.train.string_input_producer([config_dir])    # 创建输入队列，读入流中
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)  # 返回文件名和文件

    # 取出包含有image 和 label的feature对象
    features = tf.parse_single_example(example,
                                        features={'label': tf.FixedLenFeature([], tf.int64),
                                                    'data': tf.FixedLenFeature([], tf.string)})  # 将对应的内存块读为张量流
    image = tf.decode_raw(features['data'], tf.float64)  # tf.decode_raw可以将字符串解析成图像对应的像素组
    image = tf.cast(image, tf.float64)    # 解码之后转数据类型
    image = tf.reshape(image, (2600, ))
    label = tf.cast(features['label'], tf.int32)  # 类型转换
    # 随机读取数据，验证图片对应正确性
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=1,
                                                        capacity=10,
                                                        min_after_dequeue=0)

    # 开始一个会话
    with tf.Session() as sess:
        exm_images = np.zeros((num, 2600))
        exm_labels = np.zeros((num, 1))

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for count in range(num):
            image, label = sess.run([image_batch, label_batch])  # 在会话中取出image和label
            # img = image.reshape([32, 48])  # 这里要reshape因为默认一个批次处理的数据会外层嵌套一层
            # img = img.astype(np.uint8)  # PIL保存时，必须是整数
            print(image)
            print(image.shape)
            print(label)
            if num == 1:
                coord.request_stop()  # 缩进格式不对
                coord.join(threads)
                return image, label
            else:
                image = image.reshape(1536)
                # for i in range(784):
                #     # if image[i] > 127:
                #     #     image[i] = 1
                #     # else:
                #     #     image[i] = 0
                #     image[i] = image[i] / 255
                image = image / 255
                exm_images[count, :] = image
                exm_labels[count, :] = label
                if count % 1000 == 0:
                    print(count)
        coord.request_stop()  # 缩进格式不对
        coord.join(threads)
    return exm_images, exm_labels


read_tfrecord('data_train_1.tfrecords', num=1)