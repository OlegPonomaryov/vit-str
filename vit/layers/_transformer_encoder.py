"""Transformer encoder implementation as described in (Vaswani et al., 2017) and (Dosovitskiy et al., 2020)"""
import tensorflow as tf

import vit


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder (stack of encoder layers)

    :param num_layers: Number of encoder layers.
    :param model_dim: Input and output dimensionality.
    :param mha_num_heads: Number of attention heads.
    :param mlp_inner_units: MLP inner layer dimensionality.
    :param mha_key_dim: Size of each attention head for query and key (if None, model_dim // mha_num_heads is used).
    :param dropout: Rate of dropout to apply.
    :param name: String name of the layer.
    """

    def __init__(self, num_layers, model_dim, mha_num_heads, mlp_inner_units, mha_key_dim=None, dropout=0.0, name=None):
        super().__init__(name=name)

        self.encoder_layers = [TransformerEncoderLayer(model_dim, mha_num_heads, mlp_inner_units,
                                                       mha_key_dim=mha_key_dim, dropout=dropout,
                                                       name=name)
                               for _ in range(num_layers)]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """A single encoder layer.

    :param model_dim: Input and output dimensionality.
    :param mha_num_heads: Number of attention heads.
    :param mlp_inner_units: MLP inner layer dimensionality.
    :param mha_key_dim: Size of each attention head for query and key (if None, model_dim // mha_num_heads is used).
    :param dropout: Rate of dropout to apply.
    :param name: String name of the layer.
    """

    def __init__(self, model_dim, mha_num_heads, mlp_inner_units, mha_key_dim=None, dropout=0.0, name=None):
        super().__init__(name=name)

        if mha_key_dim is None:
            mha_key_dim = model_dim // mha_num_heads

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.mha = vit.layers.MultiHeadAttention(mha_num_heads, mha_key_dim)
        self.mha_dropout = tf.keras.layers.Dropout(dropout)

        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = TransformerMLP(mlp_inner_units, model_dim)
        self.mlp_dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        inputs_norm = self.layer_norm1(inputs, training=training)
        attention = self.mha(inputs_norm, inputs_norm, inputs_norm)
        attention = self.mha_dropout(attention, training=training)
        attention_residual = attention + inputs

        attention_norm = self.layer_norm2(attention_residual, training=training)
        mlp = self.mlp(attention_norm)
        mlp = self.mlp_dropout(mlp, training=training)
        mlp_residual = mlp + attention_residual

        return mlp_residual


class TransformerMLP(tf.keras.layers.Layer):
    """MLP (multilayer perceptron) - a fully connected feed-forward network located after multi-headed self-attention in
    the Transformer architecture.

    :param inner_dim: The inner-layer dimensionality.
    :param output_dim: Output dimensionality.
    :param name: String name of the layer.
    """

    def __init__(self, inner_dim, output_dim, name=None):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(inner_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
