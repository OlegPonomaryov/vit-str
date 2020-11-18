"""Tests for the CTCLoss layer."""
import tensorflow as tf

from vit.losses import CTCLoss


def test_call__padded_and_not_padded_labels():
    classes_count = 10
    padding_value = -1
    y_true = [[4, 2]]
    y_true_padded = [[4, 2, padding_value, padding_value]]
    y_pred = tf.one_hot([[classes_count, 4, classes_count, 2, 2]], classes_count + 1)
    ctc_loss = CTCLoss(true_labels_padding_value=padding_value)

    loss_without_padding = ctc_loss(y_true, y_pred)
    loss_with_padding = ctc_loss(y_true_padded, y_pred)

    # Padding of the labels shouldn't affect the loss value
    assert loss_without_padding == loss_with_padding


def test_call__output_values_are_correct():
    classes_count = 10
    padding_value = -1
    y_true = [[0, 1, 2],
              [4, 2, padding_value]]
    y_pred_good = tf.one_hot([[0, 1, classes_count, 2, classes_count],
                              [classes_count, 4, classes_count, 2, 2]], classes_count + 1)
    y_pred_bad = tf.one_hot([[9, 1, classes_count, 5, classes_count],
                             [classes_count, 4, classes_count, 5, 5]], classes_count + 1)
    ctc_loss = CTCLoss(true_labels_padding_value=padding_value)

    good_loss = ctc_loss(y_true, y_pred_good)
    bad_loss = ctc_loss(y_true, y_pred_bad)

    # A sanity check that a bad prediction produces a much higher loss than the good one
    assert bad_loss / good_loss > 1E6
