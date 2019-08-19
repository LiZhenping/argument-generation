# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:32:47 2019

@author: lizhenping
"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
#在名字为foo的命名空间内创建名字为v的变量

encoder_states=tf.get_variable("encoder_states", shape=(4,4,4,4), initializer=tf.ones_initializer())

with tf.Session() as sess:
    encoder_states.initializer.run()
    W_h= variable_scope.get_variable('v', [4,4,4,4])#  不写[1]也可以
    #sess=tf.InteractiveSession()#使用InteractiveSession函数
    W_h.initializer.run()#使用初始化器 initializer op 的 run() 方法初始化 'biases' 
    #encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
    #print(sess.run(W_h))#输出变量值
    #print(sess.run(encoder_states))
    print(sess.run(W_h))
