from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class ChangePtDetectModel(BaseModel):
    def __init__(self, config):
        super(ChangePtDetectModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None] + [self.config.step_num,self.config.fea_num])
        self.x_ = [tf.squeeze(t, [1]) for t in tf.split(self.x, self.config.step_num, 1)]

        self.y = tf.placeholder(tf.float32, shape=[None] + [self.config.step_num,self.config.fea_num])

        self.seq_len = tf.placeholder(tf.int32, shape=self.config.batch_size)
        self.y_mask = tf.placeholder(tf.float32, shape=[None] + [self.config.step_num,self.config.fea_num])


        # network_architecture
        self._rnn_cell = tf.contrib.rnn.MultiRNNCell([LSTMCell(self.config.hidden_num) for _ in range(self.config.num_hidden_layers)])
        self._rnn_cell = tf.contrib.rnn.DropoutWrapper(self._rnn_cell, output_keep_prob=self.config.dropout_keep_rate)

        with tf.variable_scope('rnn'):
            (self.z_codes, self.z_state) = tf.contrib.rnn.static_rnn(self._rnn_cell, self.x_, dtype=tf.float32, sequence_length=self.seq_len)

        with tf.variable_scope('prediction'):
            pred_weight_ = tf.Variable(tf.truncated_normal([self.config.hidden_num, self.config.fea_num], dtype=tf.float32), name='pred_weight')
            pred_bias_ = tf.Variable(tf.constant(0.1, shape=[self.config.fea_num], dtype=tf.float32), name='pred_bias')
            
            z_output_ = tf.transpose(tf.stack(self.z_codes), [1, 0, 2])
            z_weight_ = tf.tile(tf.expand_dims(pred_weight_, 0), [self.config.batch_size, 1, 1])

            self.y_ = tf.matmul(z_output_, z_weight_) + pred_bias_

        if self.config.is_training:
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.multiply(self.y_mask, tf.square(self.y-self.y_)))
                
                lr = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor, 
                                                self.config.decay_steps, self.config.decay_rate, staircase=True)
                self.optim = tf.train.AdamOptimizer(lr)
                self.train_step = self.optim.minimize(self.loss, global_step=self.global_step_tensor)


    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
