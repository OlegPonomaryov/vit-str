"""Tests for the HorizontalPatching layer."""
import numpy as np

from numpy.testing import assert_array_equal

from vit.layers import HorizontalPatching


def test_call__output_shape_is_correct():
    patch_width = 4
    samples, rows, columns, channels = 64, 32, 100, 1
    inputs = np.random.uniform(size=[samples, rows, columns, channels]).astype(np.float32)
    horizontal_patching = HorizontalPatching(patch_width, rows, channels)

    outputs = horizontal_patching.call(inputs)

    assert_array_equal(outputs.shape, [samples, columns / patch_width, rows * channels * patch_width])


def test_call__output_values_are_correct():
    patch_width = 4
    samples, rows, columns, channels = 64, 32, 100, 1
    inputs = np.random.uniform(size=[samples, rows, columns, channels]).astype(np.float32)
    horizontal_patching = HorizontalPatching(patch_width, rows, channels)

    outputs = horizontal_patching(inputs)

    for i in range(outputs.shape[1]):
        expected_patch = np.reshape(inputs[:, :, i * patch_width:(i + 1) * patch_width, :], (samples, -1))
        assert_array_equal(outputs[:, i], expected_patch)
