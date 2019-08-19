# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:44:24 2019

@author: lizhenping
"""
'''
from tensorflow.python.ops import variable_scope
import tensorflow as tf
W_h = variable_scope.get_variable("W_h", [1, 1, 4, 4])
'''

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
import numpy as np
#在名字为foo的命名空间内创建名字为v的变量

#输入
#self.emb_enc_inputs = tf.nn.embedding_lookup(embedding_encoder, self._enc_batch)



class baseModel(object):
    
    def __init__(self):
        self.rand_unif_init = \
            tf.random_uniform_initializer(-0.02,0.02,seed=123)
        self.trunc_norm_init = \
            tf.truncated_normal_initializer(stddev=0.0001)
        self.emb_enc_inputs=tf.fill([32,10,200],1.0)
        self._enc_lens=tf.fill([32],250)
        

       


    def _add_encoder(self):
        #经过两个LSTM训练，但是不明白为什么要用两个LSTM训练
        #对应的为https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
        #中的class Encoder(tf.keras.Model):函数
        cell_fw1 = tf.contrib.rnn.LSTMCell(200, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_bw1 = tf.contrib.rnn.LSTMCell(200, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_fw2 = tf.contrib.rnn.LSTMCell(200, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_bw2 = tf.contrib.rnn.LSTMCell(200, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        if 0.2 > 0.0:
            cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1,
                                                 input_keep_prob=(1 - 0.2))
            cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw1,
                                                 input_keep_prob=(1 - 0.2))
            cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw2,
                                                 input_keep_prob=(1 - 0.2))
            cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw2,
                                                 input_keep_prob=(1 - 0.2))
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw1, cell_fw2])
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw1, cell_bw2])
    

        (bi_outputs, (bi_fw_st, bi_bw_st)) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                    #emb_enc_inputs是训练输入的矩阵
                                    self.emb_enc_inputs, dtype=tf.float32,
                                    #_enc_lens是32个250
                                    sequence_length=self._enc_lens, swap_memory=True)

        # concatenate state of two layers
        bi_fw_st_conc = tf.concat(axis=2, values=[bi_fw_st[0], bi_fw_st[1]])
        bi_bw_st_conc = tf.concat(axis=2, values=[bi_bw_st[0], bi_bw_st[1]])
        print("test value")
        self.bi_fw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_fw_st_conc[0], h=bi_fw_st_conc[1])
        print("test value")
        self.bi_bw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_bw_st_conc[0], h=bi_bw_st_conc[1])
        print("test value")
        self.encoder_outputs = tf.concat(axis=2, values=bi_outputs)
        print(bi_outputs)
        print("test value")
        print("+++++++++++++++++++++++++++++")
        print("self.encoder_outputs")
        print(self.encoder_outputs)
        
        return self.encoder_outputs




    def _reduce_states(self):
        w_reduce_c = tf.get_variable('w_reduce_c',
                                 [200 * 4, 200 * 2],
                                 dtype=tf.float32, initializer=self.trunc_norm_init)
        w_reduce_h = tf.get_variable('w_reduce_h',
                                 [200 * 4, 200 * 2],
                                 dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_c = tf.get_variable('bias_reduce_c',
                                 [200 * 2], dtype=tf.float32,
                                 initializer=self.trunc_norm_init)
        bias_reduce_h = tf.get_variable('bias_reduce_h',
                                 [200 * 2], dtype=tf.float32,
                                 initializer=self.trunc_norm_init)
        #此处不理解。原因是因为自己没用个两个LSTM叠加。他为什么这么做，需要查询多一点demo
        # Apply linear layer
        #此处数学操作上仅仅是乘以一个随机矩阵。但是不理解架构上的意义。对于整个算法的数学意义
        old_c = tf.concat(axis=1, values=[self.bi_fw_st_conc.c, self.bi_bw_st_conc.c])
        old_h = tf.concat(axis=1, values=[self.bi_fw_st_conc.h, self.bi_bw_st_conc.h])
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
    #分别得到两个双向的c，h200组合成400，再拆分成原来的样子
        new_c_1, new_c_2 = tf.split(new_c, [200, 200], 1)
        new_h_1, new_h_2 = tf.split(new_h, [200, 200], 1)
        #_dec_in_state = [  2   2  32 200]分别是第一层是两个lstmstatetuple
        _dec_in_state = tuple([tf.contrib.rnn.LSTMStateTuple(new_c_1, new_h_1),
                                tf.contrib.rnn.LSTMStateTuple(new_c_2, new_h_2)])
        _dec_in_state_shape=tf.shape(_dec_in_state) 
        return _dec_in_state_shape


l1=baseModel()  
l2=l1._add_encoder()
with tf.variable_scope("encoder"):
    l2=l1._add_encoder()
    l3=l1._reduce_states()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    print("ccccccccccccccccccccccccccc")
    print(sess.run(l3))
    print("ccccccccccccccccccccccccccc")


'''    
def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    if isinstance(args, tuple):
        args = args[1]
    shapes = [a.get_shape().as_list() for a in args]
    #shapes [[32,200] [32,400]]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            #matrix shape=(600, 200)
            #values=(32,600)
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term

attention_vec_size = 400

decoder_features = linear(decoder_state, attention_vec_size, True)
#a = tf.random_normal([3, 3], mean=0.0, stddev=0.01, dtype=tf.float32)
'''  

#c=tf.Print(a,[a,a.shape,'test', a],message='Debug message:',summarize=100)
#print(c)

 
    #rint(sess.run(a.shape))
    
    
