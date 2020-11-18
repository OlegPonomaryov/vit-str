import tensorflow as tf


# Tensorflow 2 currently has MultiHeadAttention layer only in nightly builds, so until it comes to a stable release, a
# custom implementation is used. It replicates signature of the Tensorflow implementation, so that substitution could be
# easily made after its release.
class MultiHeadAttention(tf.keras.layers.Layer):
    """Implementation of multi-headed attention based on (Vaswani et al., 2017).

    :param num_heads: Number of attention heads.
    :param key_dim: Size of each attention head for query and key.
    :param name: String name of the layer.
    """

    def __init__(self, num_heads, key_dim, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.dense_query = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.dense_value = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.dense_key = tf.keras.layers.Dense(self.num_heads * self.key_dim)

        self.dense_output = None

    def call(self, query, value, key):
        if self.dense_output is None:
            self.dense_output = tf.keras.layers.Dense(key.shape[-1])

        query = self._split_heads(self.dense_query(query))
        value = self._split_heads(self.dense_value(value))
        key = self._split_heads(self.dense_key(key))

        output = self._scaled_dot_product_attention(query, value, key)

        output = self._swap_pos_head(output)
        output = tf.reshape(output, [-1, tf.shape(output)[1], self.num_heads * self.key_dim])
        output = self.dense_output(output)

        return output

    def _split_heads(self, concat_heads):
        """Splits concatenated heads."""
        concat_heads_shape = tf.shape(concat_heads)
        concat_heads = tf.reshape(
            concat_heads,
            [concat_heads_shape[0], concat_heads_shape[1], self.num_heads, -1])
        return self._swap_pos_head(concat_heads)

    def _swap_pos_head(self, x):
        """Swap position and head axes of a tensor."""
        return tf.transpose(x, [0, 2, 1, 3])

    def _scaled_dot_product_attention(self, query, value, key):
        """Calculate scaled dot product attention output."""
        attention = tf.matmul(query, key, transpose_b=True)

        key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
        attention = attention / tf.math.sqrt(key_dim)
        attention = tf.nn.softmax(attention, axis=-1)

        context = tf.matmul(attention, value)
        return context
