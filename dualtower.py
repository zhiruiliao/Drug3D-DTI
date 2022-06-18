import logging
import tensorflow as tf
import tensorflow.keras.layers as layers


class DualTower(tf.keras.models.Model):
    def __init__(self, drug_encoder, protein_encoder,
                 fc_dims, dropout_rate, **kwargs):
        super(DualTower, self).__init__(**kwargs)
        
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        
        self.fc_dims = fc_dims
        self.dropout_rate = dropout_rate
        self.pool_drug = layers.GlobalMaxPooling1D()
        self.pool_protein = layers.GlobalMaxPooling1D()
        self.fc_layers = [layers.Dense(units=_d, activation='relu') for _d in fc_dims]
        self.dropout_layers = [layers.Dropout(dropout_rate) for _d in fc_dims]
        self.final_layers = layers.Dense(units=1)
        self.num_mid_fc = len(fc_dims)
        
    def get_config(self):
        config = {
            "fc_dims": self.fc_dims,
            "dropout_rate": self.dropout_rate,
            "num_mid_fc": self.num_mid_fc
            }
        base_config = super(DualTower, self).get_config()
        config.update(base_config)
        return config
        
    def call(self, drug_data, protein_data, training):
        drug = self.drug_encoder(drug_data)
        protein = self.protein_encoder(protein_data)
        
        drug = self.pool_drug(drug)
        protein = self.pool_protein(protein)
        
        v = tf.concat([drug, protein], axis=-1)
        for i in range(self.num_mid_fc):
            v = self.dropout_layers[i](self.fc_layers[i](v), training=training)
        y = self.final_layers(v)
        return y
