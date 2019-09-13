import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
class Graph(object):

    def __init__(self, layer_num, hidden_dim, num_cls, data, knowlegde = None):
        super(Graph, self).__init__()
        self.layer = []
        self.layer_out = []
        self.knowledge = knowlegde
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.feat = data['feature']

    def setup(self):
        input = tf.expand_dims(self.feat,1)
        self.feat = tf.tile(input, multiples=[1,self.num_cls,1])
        if self.knowledge is None:
            self.knowledge = np.ones((self.num_cls, self.num_cls)).astype(np.float32) / self.num_cls
        self.knowledge = tf.Variable(self.knowledge, name ="gru-knowledge")
        self.knowledge = tf.cast(self.knowledge, dtype = tf.float32)

    def _cell(self, ind):
        input = self.feat
        num_nodes = tf.shape(input)[0]
        input_sum = tf.reduce_sum(input, 0)
        input_sum = tf.expand_dims(input_sum,0)
        input_sum = tf.tile(input_sum, multiples=[num_nodes,1,1])
        # av = tf.concat([tf.concat([tf.matmul(tf.transpose(self.knowledge, [1, 0]), (input_sum-input[i])) for i in range(num_nodes)], 0),
        #                tf.concat([tf.matmul(self.knowledge, (input_sum - input[i])) for i in range(num_nodes)], 0)], 1)
        tensor1 = tf.tensordot(tf.transpose(self.knowledge, [1, 0]), (input_sum-input), axes = [[1], [1]])
        tensor1 = tf.reshape(tensor1, [num_nodes*self.num_cls, self.hidden_dim])
        tensor2 = tf.tensordot(self.knowledge, (input_sum - input), axes = [[1], [1]])
        tensor2 = tf.reshape(tensor2, [num_nodes*self.num_cls, self.hidden_dim])
        av = tf.concat([tensor1, tensor2], 1)
        input = tf.reshape(input, [num_nodes*self.num_cls, self.hidden_dim])
        zv = tf.nn.sigmoid(slim.fully_connected(av, self.hidden_dim, scope='gru-eq3_w{}'.format(ind)) + slim.fully_connected(input, self.hidden_dim,scope='gru-eq3_u{}'.format(ind)))
        rv = tf.nn.sigmoid(slim.fully_connected(av, self.hidden_dim, scope='gru-eq4_w{}'.format(ind)) + slim.fully_connected(input, self.hidden_dim,scope='gru-eq4_u{}'.format(ind)))

        hv = tf.tanh(slim.fully_connected(av, self.hidden_dim, scope='gru-eq5_w{}'.format(ind)) + slim.fully_connected(rv * input, self.hidden_dim,scope='gru-eq5_u{}'.format(ind)))

        input = (1 - zv) * input + zv * hv

        input = tf.reshape(input, [num_nodes , self.num_cls, self.hidden_dim])
        return input


    def _iterate(self):
        self.setup()
        for i in range(self.layer_num):
            self.layer_out.append(self._cell(i))

    def get_layer_out(self):
        self._iterate()
        return self.layer_out

