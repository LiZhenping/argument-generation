"""Definition of the vanilla sequence-to-sequence model."""
import tensorflow as tf
from base_model import baseModel
from attention import attention_decoder


class VanillaSeq2seqModel(baseModel):
    """Vanilla sequence-to-sequence model with attention mechanism."""

    def _add_decoder(self):

        # define a 2-layer LSTM network for decoder
        cell1 = tf.contrib.rnn.LSTMCell(
            self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        cell2 = tf.contrib.rnn.LSTMCell(
            self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])



        #emb_arg_dec_inputs  
        #_dec_in_state
        #encoder_outputs
        #_enc_padding_mask
        # define attention on decoder
        
        #[100,32,200]
        tf.shape(self.emb_arg_dec_inputs)
       
        #(LSTMStateTuple(c=<tf.Tensor 'seq2seq_model/encoder/split:0' shape=(32, 200) dtype=float32>, h=<tf.Tensor 'seq2seq_model/encoder/split_1:0' shape=(32, 200) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'seq2seq_model/encoder/split:1' shape=(32, 200) dtype=float32>, h=<tf.Tensor 'seq2seq_model/encoder/split_1:1' shape=(32, 200) dtype=float32>))
        tf.shape(self._dec_in_state)
        #<tf.Tensor 'seq2seq_model/decoder/Shape_1:0' shape=(4,) dtype=int32>
        
        
        #<tf.Tensor 'seq2seq_model/encoder/concat_2:0' shape=(32, ?, 400) dtype=float32>
        tf.shape(self.encoder_outputs)
        
     
        #<tf.Tensor 'enc_padding_mask:0' shape=(32, ?) dtype=float32>
        tf.shape(self._enc_padding_mask)
        #<tf.Tensor 'seq2seq_model/decoder/Shape_3:0' shape=(2,) dtype=int32>
        self.arg_dec_outputs, arg_dec_out_state, arg_attn_dists = attention_decoder(
            self.emb_arg_dec_inputs, self._dec_in_state, self.encoder_outputs,
            self._enc_padding_mask, cell,
            initial_state_attention=(self.hps.mode == "decode"))
#
        self._dec_out_state = (arg_dec_out_state, [])
        self.attn_dists = (arg_attn_dists, [], [])
        return

#def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
#                      initial_state_attention=False)