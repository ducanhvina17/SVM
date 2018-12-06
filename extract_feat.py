import numpy as np
import tensorflow as tf

import vgg16
import utils, os
import cv2


PATH_DATATRAIN ='./dataset/train'
PATH_DATATEST = './dataset/test'

list_train = os.listdir(PATH_DATATRAIN)
list_test = os.listdir(PATH_DATATRAIN)

batchs = []

y_train = []
y_test = []
for item in list_train:
    # Change this path
    img = utils.load_image(os.path.join('dataset/train',item))
    img = img.reshape((1, 224, 224, 3)) # 1 image resized in 224x224x3
    batchs.append(img)
    if(item[:3]=="non"):
        y_train.append(0)
    else:
        y_train.append(1)
for item in list_test:
    # Change this path
    img = utils.load_image(os.path.join('dataset/test',item))
    img = img.reshape((1, 224, 224, 3)) # 1 image resized in 224x224x3
    batchs.append(tmp)
    if(item[:3]=="non"):
        y_test.append(0)
    else:
        y_test.append(1)

batch = np.concatenate(batchs, 0)

# np.save('y_train', np.array(y_train))
# np.save('y_test', np.array(y_test))

# Neu batch nay ma truyen vao mot luc 20 tam hinh
# thi no se tra ve fc6 kich thuoc 20

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])
        feed_dict = {images: batch}

        print('Loading model...')
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)
        print("Extracting feature...")
        fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
        print('FC6 feature: ', fc6)
        print('Number of input: ', len(fc6))
        print('Feature length of FC6: ', len(fc6[0]))
        np.save('featuredExtract', fc6)
