# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:19:25 2019

@author: lizhenping
"""


import tensorflow as tf

value = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]

print('axis=0时，拆分....')
split0, split1 = tf.split(value, [1, 1], 0)
with tf.Session() as sess:
    print(sess.run(split0))
    print("------------")
    print(sess.run(split1))
    print("------------")
    #print(sess.run(split2))

print('axis=1时，拆分....')
split0, split1, split2 = tf.split(value, [1, 2, 1], 1)
with tf.Session() as sess:
    print(sess.run(split0))
    print("------------")
    print(sess.run(split1))
    print("------------")
    print(sess.run(split2))