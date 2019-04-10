import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops

class ZeroPaddingLastDim(tf.keras.layers.Layer):
    """
    Pads the last dimension of a 3D tensor.
    """
    def __init__(self, padding=(0, 1), **kwargs):
        super(ZeroPaddingLastDim, self).__init__(**kwargs)
        self.axis = -1
        self.padding = padding
        assert len(padding) == 2

    def compute_output_shape(self, input_shape):
        if input_shape[self.axis] is not None:
            length = input_shape[self.axis] + self.padding[0] + self.padding[1]
        else:
            length = None
        return tensor_shape.TensorShape(input_shape[:self.axis] + [length])

    def call(self, inputs):
        pattern = [[0, 0], [0, 0], [self.padding[0], self.padding[1]]]
        return array_ops.pad(inputs, pattern)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ZeroPaddingLastDim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReduceSum(tf.keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        super(ReduceSum, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(ReduceSum, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        return input_shape[:self.axis] + input_shape[self.axis:]

class Reshape(tf.keras.layers.Layer):

    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape
        self.out_shape = None

    def build(self, input_shape):
        super(Reshape, self).build(input_shape)

    def call(self, x):
        reshaped_x = K.reshape(x, self.shape)
        self.out_shape = reshaped_x.get_shape()
        return reshaped_x

    def compute_output_shape(self, input_shape):
        return self.out_shape

class GGNN(tf.keras.Model):
    """
    Variant of Gated Graph Neural Net(GGNN)
    modifications:
        1. modified for undirected graph
        2. edge hidden features are used instead of weights per edge types
        3. node hidden features are updated via stacked GRUs instead of one GRU
    """
    def __init__(self,
                 node_dim,
                 node_feat_dim,
                 edge_feat_dim,
                 hidden_size,
                 output_dim,
                 num_prop=10,
                 missing_ys=False,
                 name='ggnn'):
        super(GGNN, self).__init__(name=name)
        self.node_dim = node_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_prop = num_prop
        self.missing_ys = missing_ys
        if self.node_feat_dim > self.hidden_size:
            raise ValueError('hidden_size should be greater than (or equal to) node_feat_dim .')
        num_pads = self.hidden_size - self.node_feat_dim
        self.input_node_pad = ZeroPaddingLastDim(padding=(0, num_pads))
        self.input_edge_dense = tf.keras.layers.Dense(self.hidden_size, activation='tanh')
        self.node_expand = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-3))
        self.node_edge_mul = tf.keras.layers.Lambda(lambda x: x[0] * x[1])
        self.rd_sum = ReduceSum(axis=-2)

        self.reshape_node1 = Reshape((-1, self.hidden_size))
        self.reshape_msg = Reshape((-1, 1, self.hidden_size))
        self.gru1 = tf.keras.layers.CuDNNGRU(self.hidden_size, return_sequences=True)
        self.gru2 = tf.keras.layers.CuDNNGRU(self.hidden_size, return_sequences=True)
        self.gru3 = tf.keras.layers.CuDNNGRU(self.hidden_size)
        self.reshape_node2 = Reshape((-1, self.node_dim, self.hidden_size))

        self.out_concat = tf.keras.layers.Concatenate(axis=-1)
        self.out_dense1 = tf.keras.layers.Dense(self.hidden_size, activation='sigmoid')
        self.out_dense2 = tf.keras.layers.Dense(self.hidden_size, activation='tanh')
        self.out_mul = tf.keras.layers.Multiply()
        self.out_rd_sum = ReduceSum(axis=-2)
        self.out_act = tf.keras.layers.Activation('tanh')
        self.out_dense = tf.keras.layers.Dense(self.output_dim, activation='sigmoid')

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            node_inputs = inputs[0] #dims:(v, x)
            edge_inputs = inputs[1] #dims:(v, v, e)
            node_x = self.input_node_pad(node_inputs)
            edge_x = self.input_edge_dense(edge_inputs)
            for i in range(self.num_prop):
                node_x_expanded = self.node_expand(node_x)
                msg = self.node_edge_mul([node_x_expanded, edge_x])
                msg = self.rd_sum(msg)
                node_x_reshaped = self.reshape_node1(node_x)
                msg_reshaped = self.reshape_msg(msg)
                node_x_gru = self.gru1(msg_reshaped, initial_state=node_x_reshaped)
                node_x_gru = self.gru2(node_x_gru, initial_state=node_x_reshaped)
                node_x_gru = self.gru3(node_x_gru, initial_state=node_x_reshaped)
                node_x = self.reshape_node2(node_x_gru)

            x = self.out_concat([node_x, node_inputs])
            x1 = self.out_dense1(x)
            x2 = self.out_dense2(x)
            x = self.out_mul([x1, x2])
            x = self.out_rd_sum(x)
            x = self.out_act(x)
            outputs = self.out_dense(x)
            return outputs
