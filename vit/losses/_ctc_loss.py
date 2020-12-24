"""CTC loss."""
import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    """Calculates CTC loss.

    :param true_labels_padding_value: A padding value that was used for true labels (None if there is no padding, only
        padding at the end of a sequence is supported).
    :param reduction: Reduction argument for the base tf.keras.losses.Loss class.
    :param name: Optional name for the op.
    """

    def __init__(self, true_labels_padding_value=None, reduction=tf.losses.Reduction.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.true_labels_padding_value = true_labels_padding_value

    def call(self, y_true, y_pred):
        y_true_length = self._get_length(y_true, self.true_labels_padding_value)
        y_true_length = tf.expand_dims(y_true_length, axis=1)

        y_pred_shape = tf.shape(y_pred)
        y_pred_length = tf.fill((y_pred_shape[0], 1), y_pred_shape[1])

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_length, y_true_length)
        return tf.squeeze(loss)

    def _get_length(self, sequences, padding_vale=None):
        sequences_shape = tf.shape(sequences)
        sequences_count = sequences_shape[0]
        max_sequence_length = sequences_shape[1]

        if padding_vale is None:
            return tf.fill((sequences_count,), max_sequence_length)
        else:
            not_padding = tf.not_equal(sequences, padding_vale)
            not_padding = tf.cast(not_padding, tf.int32)
            return tf.reduce_sum(not_padding, axis=1)
