import logging
import tensorflow as tf
import tensorflow.keras.layers as layers


class GatedConv1D(layers.Layer):
    """Gated Convolutional Layer.
    `output = (X conv W + b) * sigmoid(X conv V + c)`
    where `W` and `V` are convolutional kernels; * is the element-wise product.
    
    Arguments:
      kernel_size: an integer, the size of convolution kernel.
      output_dim: an integer, the dimension of output tensor.
      use_residual: a boolean, if `True`, input will be added to output.
    
    Inputs:
      X: a 3D tensor with shape: `(batch_size, input_len, input_dim)`.
    
    Outputs:
      A 3D tensor with shape: `(batch_size, input_len, output_dim)`.
    """
    def __init__(
        self,
        output_dim,
        kernel_size,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        use_residual=False,
        **kwargs):
        super(GatedConv1D, self).__init__(**kwargs)
        self.conv1d = layers.Conv1D(output_dim * 2, kernel_size, strides=strides, padding=padding,
                                    activation=None, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        
    def get_config(self):
        base_config = super(GatedConv1D, self).get_config()
        gated_conv1d_config = {
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'use_residual': self.use_residual,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer
            }
        gated_conv1d_config.update(base_config)
        return gated_conv1d_config
    
    def call(self, inputs):
        conv_1, conv_2 = tf.split(self.conv1d(inputs), num_or_size_splits=2, axis=-1)
        conv = conv_1 * tf.math.sigmoid(conv_2)
        if self.use_residual:
            return conv + inputs
        else:
            return conv


class ProteinConvEncoder(layers.Layer):
    def __init__(self, d_model, amino_vocab_size, max_amino_len, kernel_size,
                 **kwargs):
        super(ProteinConvEncoder, self).__init__(**kwargs)
        self.embed_layer = layers.Embedding(
                               input_dim=amino_vocab_size, 
                               output_dim=d_model, 
                               input_length=max_amino_len
                               )
        self.d_model = d_model
        self.amino_vocab_size = amino_vocab_size
        self.max_amino_len = max_amino_len
        
        self.conv1 = layers.Conv1D(filters=d_model, kernel_size=kernel_size, activation='relu')
        self.conv2 = layers.Conv1D(filters=d_model, kernel_size=kernel_size, activation='relu')
        self.conv3 = layers.Conv1D(filters=d_model, kernel_size=kernel_size, activation='relu')
        
        self.kernel_size = kernel_size
    
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "amino_vocab_size": self.amino_vocab_size,
            "max_amino_len": self.max_amino_len,
            "kernel_size": self.kernel_size
            }
        base_config = super(ProteinConvEncoder, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, x):
        h = self.embed_layer(x)
        z = self.conv3(self.conv2(self.conv1(h)))
        return z


class ConvGLUEncoder(layers.Layer):
    def __init__(self, d_model, kernel_size, dropout_rate=0.1, **kwargs):
        super(ConvGLUEncoder, self).__init__(**kwargs)
        self.fc = layers.Dense(units=d_model)
        
        self.conv1 = GatedConv1D(d_model, kernel_size, use_residual=True)
        self.conv2 = GatedConv1D(d_model, kernel_size, use_residual=True)
        self.conv3 = GatedConv1D(d_model, kernel_size, use_residual=True)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
    
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "kernel_size": self.kernel_size,
            "dropout_rate": self.dropout_rate
            }
        base_config = super(ConvGLUEncoder, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, x, training):
        h = self.fc(x)
        h = self.dropout1(h, training=training)
        h = self.conv1(h, training=training) * 0.7071
        
        h = self.dropout2(h, training=training)
        h = self.conv2(h) * 0.7071
        
        h = self.dropout3(h, training=training)
        h = self.conv3(h) * 0.7071
        return self.layernorm(h)
        
    
