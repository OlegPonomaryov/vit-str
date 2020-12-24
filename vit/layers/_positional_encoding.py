"""Positional encoding layer"""
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """Adds positional encoding to its inputs as described in (Vaswani et al., 2017)."""

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        positions_count = inputs_shape[1]
        encoding_dim = inputs_shape[2]

        pos = tf.expand_dims(tf.range(positions_count), axis=1)
        index = tf.range(encoding_dim)
        i = tf.expand_dims(index // 2, axis=0)

        x = tf.cast(pos, tf.float32) / tf.cast(10000**(2*i/encoding_dim), tf.float32)

        is_even = tf.less(index % 2, 1)
        positional_encoding = tf.where(is_even, tf.sin(x), tf.cos(x))

        return inputs + positional_encoding
