import logging
import tensorflow as tf
import tensorflow.keras.layers as layers

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the scaled dot product attention.

        Attention(q, k, v) = softmax(Q @ K.T / sqrt(dimension_k), axis=-1) @ V

    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
         q: Query tensor. shape = `[..., seq_len_q, dimension]`.
         k: Key tensor. shape = `[..., seq_len_k, dimension]`.
         v: Value tensor. shape = `[..., seq_len_v, dimension_v]`.
         mask: Float tensor with shape broadcastable
             to `[..., seq_len_k]`. Defaults to None.

    Returns:
        output: Output tensor. shape = `[..., seq_len_q, dimension_v]`.
        attention_weights: Attention weight tensor. shape = `[..., seq_len_q, seq_len_k]`.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.inner_dim = d_model // num_heads

        self.wq = layers.Dense(units=d_model)
        self.wk = layers.Dense(units=d_model)
        self.wv = layers.Dense(units=d_model)

        self.dense = layers.Dense(units=d_model)

    def get_config(self):
        config = {
            "Output dimension": self.d_model,
            "Num heads": self.num_heads
        }
        base_config = super(MultiHeadAttention, self).get_config()
        config.update(base_config)
        return config
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into `[num_heads, inner_dim]`,
         and transpose the result such that the shape is `[batch_size, num_heads, seq_len, inner_dim]`
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.inner_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def concatenate_heads(self, x, batch_size):
        """Merge the last dimension into `d_model`,
           and transpose the result such that the shape is `[batch_size, seq_len, d_model]`
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, q, k, v, mask, **kwargs):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        concat_attention = self.concatenate_heads(scaled_attention, batch_size)
        output = self.dense(concat_attention)

        return output, attention_weights


class DTIDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(DTIDecoderLayer, self).__init__(**kwargs)

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)
        self.dropout_3 = layers.Dropout(dropout_rate)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(DTIDecoderLayer, self).get_config()
        config.update(base_config)
        return config

    def call(self, drug, protein,
             drug_mask, protein_mask, training, **kwargs):

        attn_1, attn_weights_block_1 = self.mha_1(drug, drug, drug, drug_mask)
        attn_1 = self.dropout_1(attn_1, training=training)
        out_1 = self.layernorm_1(attn_1 + drug)

        attn_2, attn_weights_block_2 = self.mha_2(out_1, protein, protein, protein_mask)
        attn_2 = self.dropout_2(attn_2, training=training)
        out_2 = self.layernorm_2(attn_2 + out_1)

        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output, training=training)
        out_3 = self.layernorm_3(ffn_output + out_2)

        return out_3, attn_weights_block_1, attn_weights_block_2


class DTIDecoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(DTIDecoder, self).__init__(**kwargs)
        self.dec_layers = [DTIDecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Num layers": self.num_layers,
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(Decoder, self).get_config()
        config.update(base_config)
        return config

    def call(self, drug, protein, drug_mask, protein_mask,
             training, **kwargs):
        attention_weights = {}
        x = drug 
        for i in range(self.num_layers):
            x, att_block_1, att_block_2 = self.dec_layers[i](x, protein,
                                                             drug_mask, protein_mask,
                                                             training)

            attention_weights['decoder_layer_{}_block_1st'.format(i + 1)] = att_block_1
            attention_weights['decoder_layer_{}_block_2nd'.format(i + 1)] = att_block_2

        return x, attention_weights



class DrugTransformer(tf.keras.models.Model):
    def __init__(self, drug_encoder, protein_encoder,
                 num_layers, d_model, num_heads, dff, 
                 fc_units, dropout_rate=0.1, **kwargs):
        super(DrugTransformer, self).__init__(**kwargs)
        
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.fc_units = fc_units
        
        self.decoder = DTIDecoder(num_layers, d_model, num_heads, dff, dropout_rate)
        
        self.middile_layer = layers.Dense(fc_units, activation='relu')
        self.final_layer = layers.Dense(1)
    
    def call(self, drug_data, protein_data, drug_mask, protein_mask, training):
        drug = self.drug_encoder(drug_data)
        protein = self.protein_encoder(protein_data)

        dec_output, attention_weights_dict = self.decoder(
            drug, protein, drug_mask, protein_mask, training)
        norm = tf.nn.softmax(tf.norm(dec_output, axis=2), axis=1)
        h = tf.einsum("bij, bi ->bj", dec_output, norm)
        h = self.middile_layer(h)
        y = self.final_layer(h)
        return y
        
    def get_config(self):
        config = {
            "Model dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Dropout rate": self.dropout_rate,
            "Num layers": self.num_layers,
            "FC units": self.fc_units,
            "Drug encoder": self.drug_encoder,
            "Protein encoder": self.protein_encoder
        }
        base_config = super(DrugTransformer, self).get_config()
        config.update(base_config)
        return config


