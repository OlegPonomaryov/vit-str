"""Horizontal image patching layer."""
import tensorflow as tf


class HorizontalPatching(tf.keras.layers.Layer):
    """Horizontally splits an image onto flattened patches.

    :param patch_width: Width of produced patches (before flattening).
    :param name: String name of the layer.
    """

    def __init__(self, patch_width, image_height, image_channels, name=None):
        super().__init__(name=name)
        self.patch_width = patch_width
        self.image_height = image_height
        self.image_channels = image_channels

    def call(self, inputs):
        height, width, channels = self.image_height, self.patch_width, self.image_channels
        patches = tf.image.extract_patches(images=inputs, sizes=[1, height, width, channels],
                                           strides=[1, 1, width, 1], rates=[1, 1, 1, 1],
                                           padding="VALID")
        patches = patches[:, 0, :, :]  # Slicing is used since tf.squeeze() sets shape to None
        return patches
