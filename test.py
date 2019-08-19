# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:56:24 2019

@author: lizhenping
"""
import tensorflow as tf
a = tf.random_uniform_initializer(-0.2,0.2,seed=123)

b = tf.get_variable('b', [4,1], initializer=a)

t = tf.truncated_normal_initializer(stddev=0.1, seed=1)
v = tf.get_variable('v', [1], initializer=t)

with tf.Session() as sess:
    for i in range(1, 10, 1):
        sess.run(tf.global_variables_initializer())
        print(sess.run(b))
        


class test(object):
    def __init__(self):
        self.b=100
        
        
    def _add(self):
        self.a=100
    
    def _plus(self):
        b=300
        self.b=200
        t=self.a
        print("self.a",t)
        print("self.b" ,self.b,"b",b)

t=test() 
    
print(t._plus())


import tensorflow as tf
import numpy as np
 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
 
output = tf.multiply(input1, input2)
 
with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1:[3.], input2: [4.]}))
    
    
    
    
import tensorflow as tf
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print (a)
b=tf.constant(a)

with tf.Session() as sess:
    print (b)
    for x in b.eval():      #b.eval()就得到tensor的数组形式
        print (x)

    print ('a是数组',a)

    tensor_a=tf.convert_to_tensor(a)
    print ('现在转换为tensor了...',tensor_a)
    
    
import tensorflow as tf 
import numpy as np

a = tf.constant([[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]])
b=tf.reduce_sum(a, [0],keep_dims=True)
sess = tf.Session()
print(sess.run(b))
    
    
import tensorflow as tf 
import numpy as np
    
dataset = tf.data.Dataset.from_tensor_slices(np.random.randn(10,3))

dataset = dataset.batch(2, drop_remainder=True)
train_ds = dataset.take(4)
sess = tf.Session()
print(sess.run(train_ds))
#example_input_batch, example_target_batch = next(iter(dataset))

for value in train_ds:
    
    #sess = tf.Session()
   # print(sess.run(i))


import tensorflow as tf
import numpy as np
 
dataset = tf.data.Dataset.from_tensor_slices((np.random.randn(10,3)))
train_ds = dataset.take(5)
iterator = train_ds.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))



    


    
    
