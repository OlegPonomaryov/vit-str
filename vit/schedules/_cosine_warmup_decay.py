"""A cosine decay schedule with a warmup."""
import tensorflow as tf


class CosineWarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A wrapper around the CosineDecay schedule that adds a warmup feature.

    :param base_value: A value that will be achieved during warmup and than used as a starting point for decay.
    :param total_steps: Total number of steps including both warmup and decay.
    :param warmup_fraction: A fraction of total number of steps that will be used for warmup.
    """

    def __init__(self, base_value, total_steps, warmup_fraction):
        base_value = tf.cast(base_value, tf.float32)
        total_steps = tf.cast(total_steps, tf.float32)
        warmup_fraction = tf.cast(warmup_fraction, tf.float32)

        self.warmup_steps = tf.round(total_steps * warmup_fraction)
        self.warmup_rate = base_value / self.warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            base_value, total_steps - self.warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        is_warmup = tf.less(step, self.warmup_steps)
        return tf.where(is_warmup,
                        step * self.warmup_rate,
                        self.cosine_decay(step - self.warmup_steps))

    def get_config(self):
        raise NotImplementedError("CosineWarmupDecay currently doesn't have get_config() implementation")
