import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
hello=tf.constant('Hello,Tensorflow!')
print(sess.run(hello))
