# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:45:16 2019

@author: lizhenping
"""
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import numpy as np

tf.enable_eager_execution()
def load_embed_txt(embed_file, vocab):
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            if word not in vocab._word_to_id: continue
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _create_pretrained_emb_from_vocab(vocab, embed_file, dtype=tf.float32, name=None):
    trainable_tokens = vocab._word_to_id

    emb_dict, emb_size = load_embed_txt(embed_file, vocab)

    for token in trainable_tokens:
        if token not in emb_dict:
            emb_dict[token] = np.random.normal(size=(200))

    emb_mat = np.array(
        [emb_dict[token] for token in vocab._word_to_id], dtype=dtype.as_numpy_dtype())
    num_trainable_tokens = emb_mat.shape[0]
    emb_size = emb_mat.shape[1]
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    emb_mat_var = tf.get_variable(name, [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0)



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


def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                      initial_state_attention=False):
    with variable_scope.variable_scope("attention_decoder") as scope:
        # if this line fails, it's because the batch size isn't defined
        #对encoder_state 进行split分成对应每个batch_size 下的每一行的
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        
        attn_size = encoder_states.get_shape()[2].value
        print(attn_size)
        #这里不清数为什么做expend_dim
        # shape (batch_size, attn_len, 1, attn_size)
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        attention_vec_size = attn_size
        #此处是需要测试出来W_h的具体形状和数值
        #W_h(1,1,400,400)
        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])

        # shape (batch_size,attn_length,1,attention_vec_size)
        #此处自己做了实验百分之百不是仅仅的做维度变化，做了数值变化
        #代码见：
        #不明白为什么要做卷积变换
        #此处对encoder_state进行变形，此处变形用的是卷积，用encoder_state对一个W_h [shape,shape],做卷积变形。此处数值应该出现变化，而不是单单的矩阵变形。
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
        
        #此处初始化赋值的疑惑：tf.layers.conv2d中的权重初始化参数默认为None，但是就算不给参数也可以正常训练，
        #经过查看源码发现tf.layers.conv2d是从tf.keras.layer.Conv继承来的,在父类中对初始化进行了定义，
        #kernel_initializer='glorot_uniform'对卷积核参数进行均匀初始化

        #f(x)*g(t-x)在此处做卷积变换，疑惑是声明变量的数值是多少
        # Get the weight vectors v and w_c (w_c is for coverage)
        #此处的V是直接定义的一个矩阵。但是没有进行权重初始化
        #标准的写法不是这种，是利用生成一个神经网络层进行初始化，但是此处应该雇佣纠结初始化数值问题，用的nn_ops库中的矩阵在initializer的时候能够自动初始化。相当于同tf例子中的作用一样，不一样的函数表达式
        v = variable_scope.get_variable("v", [attention_vec_size])
        #此处是使用一个BahdanauAttention算法来分配注意力
        def attention(decoder_state):
            with variable_scope.variable_scope("attention"):

                # Pass the decoder state through a linear layer
                # shape (batch_size, attention_vec_size)
                #linear的作用做a*decode_state+bias 这里面感觉自己的理解是错误的？
                #decoder_features=[32,200]
                #decoder_state=decoder_state
                #linear(args, output_size, bias, bias_start=0.0, scope=None):
                #此处没搞明白参数的对应关系。
                #
                decoder_state
                attention_vec_size
                print(decoder_state)
                #此处linear没做审什么变动知识做了list tuple检查，如果decoder_state形式正确，decoder没任何变化
                decoder_features = linear(decoder_state, attention_vec_size, True)

                # reshape to (batch_size, 1, 1, attention_vec_size)
                #此处一直没明白为什么要reshape，此处运行是要大量时间，是为了？
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize


                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)求出来score数值
                #score = FC(tanh(FC(EO) + FC(H)))
                #FC = Fully connected (dense) layer
                #EO = Encoder output
                #H = hidden state
                #X = input to the decoder
                #e=score
                #V=400
                e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features), [2, 3])
                # Calculate attention distribution
                attn_dist = masked_attention(e)

                # Calculate the context vector from attn_dist and encoder_states
                #对attn_dist进行变形 由   变成   并乘以输入的encoder*state得到一个注意力分配模型
                #此处疑惑的是矩阵的形状 返回的atten_dist应该没用
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2])  # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist

        outputs = []
        attn_dists = []
        state = initial_state
        context_vector = array_ops.zeros([batch_size, attn_size])

        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step 
            # so that we can pass it through a linear layer with 
            # this step's input to get a modified version of the input
            context_vector, _ = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                    context_vector, attn_dist = attention(state)
            else:
                context_vector, attn_dist = attention(state)
            attn_dists.append(attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector, 
            # and pass them through a linear layer
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists
    

class baseModel(object):
    """Base class for seq2seq models."""
    def __init__(self):
        #hps传入的是parser args的数值，用于加载输入参数的数值
        #在此处声明类中需要用的变量
        
        self.arg_dec_outputs = None
        self._dec_out_state = None
        self.attn_dists = None
        
        
        
        
        def _add_embedding(self):
        #此处是加载glove/glove.6B.200d.txt数据集
        if os.path.exists(self.hps.embed_path):
            embedding_encoder = _create_pretrained_emb_from_vocab(
                self._src_vocab, self.hps.embed_path, name="embedding_src")
            embedding_decoder = _create_pretrained_emb_from_vocab(
                self._tgt_vocab, self.hps.embed_path, name="embedding_tgt")
        else:
            #如果未加载自动生成一个同样大小的矩阵，随机生成
            embedding_encoder = tf.get_variable('embedding_src',
                                                [self._src_vocab.size(), self.hps.emb_dim],
                                                dtype=tf.float32,
                                                initializer=self.trunc_norm_init)
           
        #利用look_up查找对应的单词在表中的位置做单词和ebededing的映射
        self.emb_enc_inputs = tf.nn.embedding_lookup(embedding_encoder, self._enc_batch)
        self.emb_arg_dec_inputs = [tf.nn.embedding_lookup(embedding_decoder, x)
                                   for x in tf.unstack(self._arg_dec_batch, axis=1)]
        if self.hps.model in ["sep_dec", "shd_dec"]:
            self.emb_kp_dec_inputs = [tf.nn.embedding_lookup(embedding_decoder, x)
                                      for x in tf.unstack(self._kp_dec_batch, axis=1)]




        #print("  Time to a
    def _attention_decoder(self):
        
        self.rand_unif_init = \
                tf.random_uniform_initializer(-0.02,
                                              0.02,
                                              seed=123)
                # define a 2-layer LSTM network for decoder
        cell1 = tf.contrib.rnn.LSTMCell(
            200, state_is_tuple=True, initializer=self.rand_unif_init)
        cell2 = tf.contrib.rnn.LSTMCell(
            200, state_is_tuple=True, initializer=self.rand_unif_init)
        cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
        
        embedding_decoder = tf.get_variable('embedding_tgt',
                                                [self._tgt_vocab.size(), self.hps.emb_dim],
                                                dtype=tf.float32,
                                                initializer=self.trunc_norm_init)
        
        emb_arg_dec_inputs = tf.Variable(tf.fill([100,32,200], 0.4)) 
        emb_arg_dec_inputs = tf.convert_to_tensor(emb_arg_dec_inputs)
        
        _dec_in_state = tf.Variable(tf.fill([4,100], 0.6))  
        _dec_in_state = tf.convert_to_tensor(_dec_in_state) 
        encoder_outputs = tf.Variable(tf.fill([32,100,400], 0.7))
        encoder_outputs = tf.convert_to_tensor(encoder_outputs) 
        _enc_padding_mask = tf.Variable(tf.fill([32,100], 0.8))
               
        enc_padding_mask = np.ones((32, 250), dtype=np.float32)
  
     

        enc_padding_mask = tf.convert_to_tensor(_enc_padding_mask)
        print("this is a test")
   

        
        self.arg_dec_outputs, arg_dec_out_state, arg_attn_dists = attention_decoder(
            self.emb_arg_dec_inputs, _dec_in_state, encoder_outputs,
            enc_padding_mask, cell,False)
        
        

def main():
    
    
    l1 = baseModel()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(l1._attention_decoder())
     
      
  
    
        
  
        
        
if __name__ == '__main__':
  main()