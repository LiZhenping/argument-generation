import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

#此函数用于做attention模型
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
        #这里数值其实是变化了传统公式里面使用的一层神经网络这里换成了卷积神经网络
       
        #其中原来公式
        #FC = Fully connected (dense) layer
        #EO = Encoder output
        #FC(EO)
        #这里FC是卷积神经网络
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
                
                decoder_state
                attention_vec_size
                print(decoder_state)
                #此处对应的公式为：
                #这里是FC(H)
                #其中H是hidden state 是decoder的hidden state
                #这里做的是一个神经网络运算
                decoder_features = linear(decoder_state, attention_vec_size, True)

                # reshape to (batch_size, 1, 1, attention_vec_size)
                #此处一直没明白为什么要reshape，应该是为了以后运算维度对齐
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    #这里的操作是为了求真实的softmax ，因为句子做了padding，
                    attn_dist *= enc_padding_mask  # apply mask
                    #axis = 1相当于把每个向量内部的进行加和并减少一个维度
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
                #[2, 3] 是指的在分别2，3维度上进行加和现在第二维度，再在第三维度上进行加和
                e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features), [2, 3])
                # Calculate attention distribution
                #在这里分别计算注意力
                attn_dist = masked_attention(e)

                # Calculate the context vector from attn_dist and encoder_states
                #对attn_dist进行变形 由   变成   并乘以输入的encoder*state得到一个注意力分配模型
                #此处疑惑的是矩阵的形状 返回的atten_dist应该没用
                #这里吧attn_dist进行reshape操作，分配每个encoder的注意力
                #attn_dist, [batch_size, -1, 1, 1] 为reshape后的每个encoder的注意力
                #encoder = values attn_dist=attention_weight
                #tf.reduce_sum(attention_weights * values,axis=1)
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
        #这里为false不经过这个if
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
            #output , state = self.gru(x)
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
            #x=self.fc(output)
            with variable_scope.variable_scope("AttnOutputProjection"):
                # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists

def dual_attention_shared_decoder(decoder_inputs, initial_state, encoder_states,
                      dec_2_decoder_states, enc_padding_mask, cell, dec_2_padding_mask,
                      initial_state_attention=False, reuse=False, use_dual=False, 
                      pad_last_state=False):

    if reuse:
        reuse = tf.AUTO_REUSE

    with variable_scope.variable_scope("dual_attention_decoder", reuse=reuse) as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        enc_padding_mask = tf.reshape(enc_padding_mask, [batch_size, -1])
        shared_params = flat_attention_init(encoder_states, enc_padding_mask, reuse=reuse)
        attention = flat_attention

        dual_attn_size = dec_2_decoder_states.get_shape()[2].value
        if use_dual:
            dual_shared_params = flat_attention_init(dec_2_decoder_states, dec_2_padding_mask, 
                    use_dual=True, scope="dual_Attention")

        def stop_when(time, unused_state, unused_last_states, unused_context_vector, 
                unused_dual_context_vector, unused_decoder_states,
                unused_attn_dists, unused_dual_attn_dists, unused_outputs):
            return tf.less(time, tf.constant(len(decoder_inputs)))

        def one_step(time, state, last_states, context_vector, dual_context_vector, 
                decoder_states, attn_dists, dual_attn_dists, outputs):
            inp = tf.gather(decoder_inputs, time)
            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if use_dual:
                x = linear([inp] + [context_vector] + [dual_context_vector], input_size, True,
                           scope='dual_input_transform')
            else:
                x = linear([inp] + [context_vector], input_size, True, scope='input_transform')
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)
            decoder_states = decoder_states.write(time, cell_output)
            last_states = last_states.write(time, state)

            # Run the attention mechanism.
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True):
                context_vector, attn_dist = attention(state, shared_params)

                if use_dual:
                    dual_context_vector, dual_attn_dist = flat_attention(state, dual_shared_params,
                                                                            scope="Dual_Attention")

            attn_dists = attn_dists.write(time, attn_dist)
            if use_dual:
                dual_attn_dists = dual_attn_dists.write(time, dual_attn_dist)

            if use_dual:
                output = linear([cell_output] + [context_vector] + [dual_context_vector], cell.output_size, True,
                                scope="DualAttnOutputProjection")
            else:
                output = linear([cell_output] + [context_vector], cell.output_size, True, scope="AttnOutputProjection")
            outputs = outputs.write(time, output)
            return (
            time + 1, state, last_states, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs)

        initial_time = tf.constant(0)
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_dual_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_decoder_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_context_vector = array_ops.zeros([batch_size, attn_size])
        initial_context_vector.set_shape([None, attn_size])
        initial_dual_context_vector = array_ops.zeros([batch_size, dual_attn_size])
        initial_dual_context_vector.set_shape([None, dual_attn_size])
        initial_last_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))

        vec, _ = attention(initial_state, shared_params)
        initial_context_vector = tf.cond(initial_state_attention,
                                         lambda: (vec),
                                         lambda: (initial_context_vector))
        if use_dual:
            initial_dual_context_vector = tf.cond(initial_state_attention,
                                                  lambda: flat_attention(initial_state,
                                                      dual_shared_params,
                                                      scope="Dual_Attention")[0],
                                                  lambda: initial_dual_context_vector)

        time, last_state, last_states, context_vector, dual_context_vector,
        decoder_states, attn_dists, dual_attn_dists, outputs = tf.while_loop(stop_when, one_step,
                          loop_vars=[initial_time, initial_state, initial_last_states,
                                     initial_context_vector, initial_dual_context_vector,
                                     initial_decoder_states,
                                     initial_attn_dists, initial_dual_attn_dists, initial_outputs],
                                     parallel_iterations=32, swap_memory=True)

        outputs = tf.unstack(outputs.stack(), num=len(decoder_inputs))
        decoder_states = tf.stack(
                tf.unstack(decoder_states.stack(), num=len(decoder_inputs)), axis=1)
        attn_dists = tf.unstack(attn_dists.stack(), num=len(decoder_inputs))
        if use_dual:
            dual_attn_dists = tf.unstack(dual_attn_dists.stack(), num=len(decoder_inputs))
        else:
            dual_attn_dists = None
        last_states = tf.unstack(last_states.stack(), num=len(decoder_inputs))
        if pad_last_state:
            padding_inds = tf.cast(tf.reduce_sum(dec_2_padding_mask, axis=1), tf.int32) - 1
            tmp_1 = tf.unstack(last_states, axis=3)
            tmp_2 = tf.unstack(padding_inds)

            tmp_3 = tmp_1[0]
            tmp_4 = tf.transpose(tmp_3, perm=[1, 0, 2, 3])
            tmp_5 = tmp_4[0]

            last_state_0 = [tf.transpose(x, perm=[1, 0, 2, 3])[0][ind] for x, ind in
                            zip(tf.unstack(last_states, axis=3), tf.unstack(padding_inds))]

            last_state_0 = tf.transpose(last_state_0, perm=[1, 0, 2])
            last_state_0 = tf.contrib.rnn.LSTMStateTuple(last_state_0[0], last_state_0[1])

            last_state_1 = [tf.transpose(x, perm=[1, 0, 2, 3])[1][ind] for x, ind in
                            zip(tf.unstack(last_states, axis=3), tf.unstack(padding_inds))]

            last_state_1 = tf.transpose(last_state_1, perm=[1, 0, 2])
            last_state_1 = tf.contrib.rnn.LSTMStateTuple(last_state_1[0], last_state_1[1])

            last_state = (last_state_0, last_state_1)

    return outputs, last_state, decoder_states, attn_dists, dual_attn_dists

def dual_attention_decoder(decoder_inputs, initial_state, encoder_states, dec_2_decoder_states, 
                           enc_padding_mask, cell, dec_2_padding_mask, 
                           initial_state_attention=False, reuse=False, use_dual=False):
    with variable_scope.variable_scope("dual_attention_decoder", reuse=reuse) as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value
        enc_padding_mask = tf.reshape(enc_padding_mask, [batch_size, -1])
        shared_params = flat_attention_init(encoder_states, enc_padding_mask, reuse=reuse)
        attention = flat_attention

        dual_attn_size = dec_2_decoder_states.get_shape()[2].value
        if use_dual:
            dual_shared_params = flat_attention_init(dec_2_decoder_states, dec_2_padding_mask,
                                                    use_dual=True, scope="Dual_Attention")

        def stop_when(time, unused_state, unused_context_vector, unused_dual_context_vector,
                      unused_decoder_states, unused_attn_dists, unused_dual_attn_dists,
                      unused_outputs):
            return tf.less(time, tf.constant(len(decoder_inputs)))

        def one_step(time, state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists,
                     outputs):
            inp = tf.gather(decoder_inputs, time)
            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if use_dual:
                x = linear([inp] + [context_vector] + [dual_context_vector], input_size, True,
                           scope='dual_input_transform')
            else:
                x = linear([inp] + [context_vector], input_size, True, scope='input_transform')
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)
            decoder_states = decoder_states.write(time, cell_output)

            # Run the attention mechanism.
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True):
                context_vector, attn_dist = attention(state, shared_params)

                if use_dual:
                    dual_context_vector, dual_attn_dist = flat_attention(state, dual_shared_params,
                                                                            scope="Dual_Attention")

            attn_dists = attn_dists.write(time, attn_dist)
            if use_dual:
                dual_attn_dists = dual_attn_dists.write(time, dual_attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            if use_dual:
                output = linear([cell_output] + [context_vector] + [dual_context_vector], cell.output_size, True,
                                scope="DualAttnOutputProjection")
            else:
                output = linear([cell_output] + [context_vector], cell.output_size, True, scope="AttnOutputProjection")
            outputs = outputs.write(time, output)
            return (
            time + 1, state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs)

        initial_time = tf.constant(0)
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_dual_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_decoder_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_context_vector = array_ops.zeros([batch_size, attn_size])
        initial_context_vector.set_shape([None, attn_size])
        initial_dual_context_vector = array_ops.zeros([batch_size, dual_attn_size])
        initial_dual_context_vector.set_shape(
            [None, dual_attn_size])


        vec, _ = attention(initial_state, shared_params)
        initial_context_vector = tf.cond(initial_state_attention, \
                                                           lambda: (vec), \
                                                           lambda: (initial_context_vector))
        if use_dual:
            initial_dual_context_vector = tf.cond(initial_state_attention, \
                                                  lambda: flat_attention(initial_state, dual_shared_params,
                                                                         scope="Dual_Attention")[0], \
                                                  lambda: initial_dual_context_vector)

        time, last_state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs = \
            tf.while_loop(stop_when, one_step, \
                          loop_vars=[initial_time, initial_state, initial_context_vector, initial_dual_context_vector,
                                     initial_decoder_states, \
                                     initial_attn_dists, initial_dual_attn_dists, initial_outputs], \
                          parallel_iterations=32, swap_memory=True)

        outputs = tf.unstack(outputs.stack(), num=len(decoder_inputs))
        decoder_states = tf.stack(tf.unstack(decoder_states.stack(), num=len(decoder_inputs)), axis=1)
        attn_dists = tf.unstack(attn_dists.stack(), num=len(decoder_inputs))
        if use_dual:
            dual_attn_dists = tf.unstack(dual_attn_dists.stack(), num=len(decoder_inputs))
        else:
            dual_attn_dists = None

    return outputs, last_state, decoder_states, attn_dists, dual_attn_dists

#linear(decoder_state, attention_vec_size, True)
    #其实这里就是个神经网络层。相当于自己手写了神经网络的矩阵乘法l
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
    #初始化BP矩阵
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            #matrix shape=(600, 200)
            #values=(32,600)
            #矩阵乘法，做神经网络运算
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        #增加偏置
    return res + bias_term


def flat_attention_init(encoder_states, enc_padding_mask, use_dual=False, reuse=False, scope=None):
    with variable_scope.variable_scope(scope or "Attention"):
        batch_size = encoder_states.get_shape()[0].value  # if this line fails, it's because the batch size isn't defined
        attn_size = encoder_states.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined
        attention_vec_size = attn_size

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        if use_dual:
            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_d", [1, 1, attn_size, attention_vec_size])
            # Get the weight vectors v
            v = variable_scope.get_variable("v_d", [attention_vec_size])

            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")

        else:
            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            # Get the weight vectors v
            v = variable_scope.get_variable("v", [attention_vec_size])

            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

        return (batch_size, attn_size, encoder_states, attention_vec_size, encoder_features, v, enc_padding_mask, None)


def flat_attention(decoder_state, shared_params, scope=None):
    (batch_size, _, encoder_states, attention_vec_size, encoder_features, v, enc_padding_mask, _) = shared_params

    with variable_scope.variable_scope(scope or "attention"):
        decoder_features = linear(decoder_state, attention_vec_size, True,
                                  scope="decoder_features")  # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                          1)  # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length) or (batch_size, num_heads, attn_length) if multihead
            attn_dist *= enc_padding_mask  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=-1,
                                        keep_dims=True)  # shape (batch_size,1) or (batch_size,num_heads,1) if multihead
            return attn_dist / masked_sums


        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [-1, -2])

        # Calculate attention distribution
        attn_dist = masked_attention(e)


        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                                                 [1, 2])  # shape (batch_size, attn_size).

    return context_vector, attn_dist
