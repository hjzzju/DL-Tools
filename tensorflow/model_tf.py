class BaseModel(object):
    def __init__(self, **kwargs):
        # get the name or logging used in the following
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        # store the data
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        # The following four is the results to be fetched in tensorflow graph
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0
    
    # build the network
    def _build(self):
        raise NotImplementedError

    # wrapper for _build() and run the graph
    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        # activations to store data
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        # tf.GraphKeys.GLOBAL_VARIABLES means all variables in the graph
        # tf.get_collection means get the keys and return a collection
        # scope filters out the name of keys of the collection, if None, all keys in the return
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build the nodes
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)
        # calculate gradient with respect to input, only for dense model
        # self.grads = tf.gradients(self.loss, self.inputs)[0]  # does not work on sparse vector

    # return the results
    def predict(self):
        pass
    
    # get the loss and store in self.loss
    def _loss(self):
        raise NotImplementedError
    
    # get the accuracy and store in self.accuracy
    def _accuracy(self):
        raise NotImplementedError

    # save the model
    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    # load the model
    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

# An example for using the base Model
# class GCN_dense_mse(Model_dense):
#     def __init__(self, placeholders, input_dim, **kwargs):
#         super(GCN_dense_mse, self).__init__(**kwargs)

#         self.inputs = placeholders['features']
#         self.input_dim = input_dim
#         # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
#         self.output_dim = placeholders['labels'].get_shape().as_list()[1]
#         self.placeholders = placeholders

#         # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
#         self.build()

#     def _loss(self):
#         # Weight decay loss
#         for i in range(len(self.layers)):
#             for var in self.layers[i].vars.values():
#                 self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

#         # Cross entropy error
#         self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
#                                    self.placeholders['labels_mask'])

#     def _accuracy(self):
#         self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
#                                       self.placeholders['labels_mask'])

#     def lrelu(x, leak=0.2, name="lrelu"):
#         return tf.maximum(x, leak * x)

#     def _build(self):
#         self.layers.append(GraphConvolution(input_dim=self.input_dim,
#                                             output_dim=FLAGS.hidden1,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.maximum(x, 0.2 * x),
#                                             dropout=False,
#                                             sparse_inputs=False,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
#                                             output_dim=FLAGS.hidden2,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.maximum(x, 0.2 * x),
#                                             dropout=False,
#                                             sparse_inputs=False,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
#                                             output_dim=FLAGS.hidden3,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.maximum(x, 0.2 * x),
#                                             dropout=False,
#                                             sparse_inputs=False,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
#                                             output_dim=FLAGS.hidden4,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.maximum(x, 0.2 * x),
#                                             dropout=False,
#                                             sparse_inputs=False,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden4,
#                                             output_dim=FLAGS.hidden5,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.maximum(x, 0.2 * x),
#                                             dropout=True,
#                                             sparse_inputs=False,
#                                             logging=self.logging))

#         self.layers.append(GraphConvolution(input_dim=FLAGS.hidden5,
#                                             output_dim=self.output_dim,
#                                             placeholders=self.placeholders,
#                                             act=lambda x: tf.nn.l2_normalize(x, dim=1),
#                                             dropout=True,
#                                             logging=self.logging))

#     def predict(self):
#         return self.outputs