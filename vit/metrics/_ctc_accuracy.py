"""CTC accuracy."""
import tensorflow as tf


class CTCAccuracy(tf.keras.metrics.Mean):
    """Calculates how often a predicted text fully matches the corresponding true text.

    :param true_labels_padding_value:  A padding value that was used for true labels (None if there is no padding, only
        padding at the end of a sequence is supported).
    :param name: A string name of the metric instance.
    """

    # An unknown character token returned by ctc_decode(), which also serves as a padding constant
    DECODE_PADDING_VALUE = -1

    def __init__(self, true_labels_padding_value=None, name="accuracy"):
        super().__init__(name=name)
        self.true_labels_padding_value = true_labels_padding_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)

        y_pred_length = tf.fill((y_pred_shape[0],), y_pred_shape[1])
        y_pred = tf.keras.backend.ctc_decode(y_pred, y_pred_length)[0][0]
        y_pred_shape = y_pred_shape[:-1]  # y_pred shape after the decoding loses the last dimension (classes count)

        # If y_true is padded, update it to use the same padding constant as y_pred_decoded
        if self.true_labels_padding_value is not None and \
           self.true_labels_padding_value != CTCAccuracy.DECODE_PADDING_VALUE:
            y_true = tf.where(tf.not_equal(y_true, self.true_labels_padding_value),
                              y_true, tf.fill(y_true_shape, CTCAccuracy.DECODE_PADDING_VALUE))

        # Pad y_true or y_pred so that they have the same max sequence length
        if y_true_shape[1] < y_pred_shape[1]:
            y_true = tf.pad(y_true, paddings=[[0, 0], [0, y_pred_shape[1] - y_true_shape[1]]],
                            constant_values=CTCAccuracy.DECODE_PADDING_VALUE)
            y_true_shape = y_pred_shape
        elif y_pred_shape[1] < y_true_shape[1]:
            y_pred = tf.pad(y_pred, paddings=[[0, 0], [0, y_true_shape[1] - y_pred_shape[1]]],
                            constant_values=CTCAccuracy.DECODE_PADDING_VALUE)
            y_pred_shape = y_true_shape

        # ctc_decode() returns y_pred of int64 type and if y_true is int32, that will cause an error in tf.equal(), so
        # we need to cast y_pred to the same type
        if y_pred.dtype != y_true.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)

        matching_labels_count = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32), axis=1)
        # A prediction considered correct if all its labels match the corresponding true sequence
        correct_predictions = tf.equal(matching_labels_count, y_true_shape[1])

        super().update_state(tf.cast(correct_predictions, tf.int32), sample_weight)
