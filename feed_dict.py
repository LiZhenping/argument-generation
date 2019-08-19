# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:30:59 2019

@author: lizhenping
"""
#demo1
#传值例子
import tensorflow as tf
import numpy as np
fa=tf.placeholder(tf.float32,shape=[2,1])

fb=tf.placeholder(tf.float32,shape=[1,2])

def test(a,b):
    biases=tf.Variable(tf.zeros([1])+0.1)
    Wx_plus_b=tf.matmul(fa,fb)+biases
    tf.print(Wx_plus_b)
    print(Wx_plus_b)
    
    return Wx_plus_b
    
l1=test(fa,fb)    



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a=np.random.randint(1,2,(2,1))
    b=np.random.randint(1,2,(1,2))

    
    print(sess.run(l1,feed_dict={fa:a,fb:b}))
    
'''
demo2
import tensorflow as tf  

# 设计Graph  

x1 = tf.placeholder(tf.int16)  

x2 = tf.placeholder(tf.int16)  

y = tf.add(x1, x2)  

# 用Python产生数据  

li1 = [2, 3, 4]  

li2 = [4, 0, 1]  

# 打开一个session --> 喂数据 --> 计算y  

with tf.Session() as sess:  
    print(sess.run(y, feed_dict={x1: li1, x2: li2}))
'''