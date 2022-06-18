import logging
import tensorflow as tf
import tensorflow.keras.layers as layers

logging.getLogger('tensorflow').setLevel(logging.ERROR)

class GraphConv(layers.Layer):
    """Graph convolution layer.
    Xnew = activation(AXW + b) * Mask
        Args:
            d_model: int, the output dimension.
            use_bias: bool, whether the bias is used.
            activation: str or callable, the activation function.

        Inputs:
            a: Adjacency matrix A. shape = `(batch_size, n, n)`
            x: Input matrix X. shape = `(batch_size, n, d_input)`

        Outputs:
            xnew: Updated feature matrix X_{i+1}. shape = `(batch_size, n, d_model)`
    """

    def __init__(self, d_model, use_bias=True, activation=None, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.d_model = d_model
        self.use_bias = use_bias
        self.activation = activation
        self.dense = layers.Dense(units=d_model, activation=activation, use_bias=use_bias)

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "use_bias": self.use_bias,
            "activation": self.activation
        }
        base_config = super(GraphConv, self).get_config()
        config.update(base_config)
        return config

    def call(self, a, x):
        ax = tf.matmul(a, x)
        z = self.dense(ax)
        return z


class Drug3DEncoder(layers.Layer):
    def __init__(self, d_model, n_layers_2d=2, n_layers_3d=1, **kwargs):
        super(Drug3DEncoder, self).__init__(**kwargs)
        self.gcn_2d = []
        for i in range(n_layers_2d - 1):
            self.gcn_2d.append(GraphConv(d_model, 'relu'))
        self.gcn_2d.append(GraphConv(d_model))
        
        self.gcn_3d = []
        for i in range(n_layers_3d - 1):
            self.gcn_3d.append(GraphConv(d_model, 'relu'))
        self.gcn_3d.append(GraphConv(d_model))
        
        self.fc = layers.Dense(units=d_model)
        self.d_model = d_model
        self.n_layers_2d = n_layers_2d
        self.n_layers_3d = n_layers_3d
        
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "n_layers_2d": self.n_layers_2d,
            "n_layers_3d": self.n_layers_3d
            }
        base_config = super(Drug3DEncoder, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, inputs):
        a, s, x = inputs
        
        ha = x
        for i in range(self.n_layers_2d):
            ha = self.gcn_2d[i](a, ha)
            
        hs = x
        for i in range(self.n_layers_3d):
            ha = self.gcn_3d[i](s, hs)
        
        h = tf.concat([ha, hs], axis=-1)
        z = self.fc(h)
        return z


class Drug2DEncoder(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(Drug2DEncoder, self).__init__(**kwargs)
        self.gcn = GraphConv(d_model)
        self.d_model = d_model
        
    def get_config(self):
        config = {
            "d_model": self.d_model
            }
        base_config = super(Drug2DEncoder, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, inputs):
        a, x = inputs
        h = self.gcn(a, x)
        return h
