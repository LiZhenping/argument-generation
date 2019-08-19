# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:57:16 2019

@author: lizhenping
"""
#关于作用域的定义
#关于TensorFlow 如何通过return 运行

import tensorflow as tf
from tensorflow.python.ops import variable_scope

        
        
def attention_decoder(emb_arg_dec_inputs, _dec_in_state, encoder_outputs, _enc_padding_mask, cell,initial_state_attention=False):
    with variable_scope.variable_scope("attention_decoder") as scope:  
        print(emb_arg_dec_inputs)
        print(_dec_in_state)
        print(encoder_outputs)
        print(_enc_padding_mask)
        print(cell)
        print(initial_state_attention)
       
 

class baseModel(object):
    """Base class for seq2seq models."""
    def __init__(self):
        #hps传入的是parser args的数值，用于加载输入参数的数值
        #在此处声明类中需要用的变量
        
        self.arg_dec_outputs = None
        self._dec_out_state = None
        self.attn_dists = None

    def __call__(self):
       
        
        self._add_placeholder()
        self._add_model()
        #print("  Time to add model: %i seconds" % (time.time() - t0))
   
        
        return
    def _add_embedding(self):
        self.emb_arg_dec_inputs = 1
    
    def _reduce_states(self):
        self._dec_in_state = 1
    def _add_encoder(self):
        self.encoder_outputs = 1
        
    def _add_placeholder(self):
        self._enc_padding_mask = tf.fill([32,200],9.3)
    def _add_model(self): 
        with tf.variable_scope("seq2seq_model"):
            self.rand_unif_init = \
                tf.random_uniform_initializer(-0.02,
                                              0.02,
                                              seed=123)
            with tf.variable_scope('embedding'):
                self._add_embedding()
            with tf.variable_scope("encoder"):
                self._add_encoder()
                self._reduce_states()

            

       
class VanillaSeq2seqModel(baseModel):

    def _add_decoder(self):
        

        # define a 2-layer LSTM network for decoder
        cell1 = tf.contrib.rnn.LSTMCell(
            200, state_is_tuple=True, initializer=self.rand_unif_init)
        cell2 = tf.contrib.rnn.LSTMCell(
            200, state_is_tuple=True, initializer=self.rand_unif_init)
        cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])

        # define attention on decoder
        _ = attention_decoder(
            self.emb_arg_dec_inputs, self._dec_in_state, self.encoder_outputs,
            self._enc_padding_mask, cell,
            initial_state_attention=(True))
#
        print("test_end")
        
        return

def cond(a, b, c): 
    return a<100 
 

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    print(sess.run([a, b, c])) #[5, 6, 7]


def main():
    
    model = VanillaSeq2seqModel()
    

    model()
 
    with tf.Session() as sess:
        print(sess.run(model._add_decoder()))
        
        
if __name__ == '__main__':
  main()
        

