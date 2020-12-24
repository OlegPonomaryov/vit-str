"""Tools for the IIIT 5K-word dataset dataset."""
import tensorflow as tf
from mat4py import loadmat
from pathlib import Path
import string
import os


import datasets


IIIT5K_CHARS = list(string.digits + string.ascii_uppercase)


def load_iiit5k(path):
    """Loads the IIIT 5K-word dataset as tf.data.Dataset objects.

    :param path: Path to the root directory of the dataset
    :return: Train and test datasets
    """
    path = Path(path)

    rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=datasets.IMAGE_SCALE, offset=datasets.IMAGE_OFFSET)

    filename_to_text_train = load_filename_to_text_dict(path, "traindata")
    filename_to_text_test = load_filename_to_text_dict(path, "testdata")

    labels = tf.range(len(IIIT5K_CHARS), dtype=tf.int32)
    char_to_label = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(IIIT5K_CHARS, labels), -1)

    train_ds = build_dataset(path / "train", rescaling, filename_to_text_train, char_to_label)
    test_ds = build_dataset(path / "test", rescaling, filename_to_text_test, char_to_label)

    return train_ds, test_ds


def load_filename_to_text_dict(dataset_path, name):
    filename_to_text = loadmat((dataset_path / f"{name}.mat").as_posix())[name]
    filename_to_text = {Path(path).name: text for path, text in zip(filename_to_text["ImgName"],
                                                                    filename_to_text["GroundTruth"])}
    return dict_to_hash_table(filename_to_text)


def dict_to_hash_table(dictionary, default_value=""):
    keys = list(dictionary.keys())
    values = [dictionary[key] for key in keys]
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value)


def build_dataset(path, rescaling, filename_to_text, char_to_label):
    files_ds = tf.data.Dataset.list_files((path / "*.png").as_posix(), shuffle=False)
    return files_ds.map(
        lambda image_path: process_path(image_path, rescaling, filename_to_text, char_to_label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def process_path(path, rescaling, filename_to_text, char_to_label):
    image = load_image(path, rescaling)
    labels = get_labels(path, filename_to_text, char_to_label)
    return image, labels


def load_image(path, rescaling):
    image_data = tf.io.read_file(path)
    image = tf.image.decode_png(image_data, channels=datasets.IMAGE_CHANNELS)
    image = tf.image.resize(image, [datasets.IMAGE_HEIGHT, datasets.IMAGE_WIDTH])
    image = rescaling(image)
    return image


def get_labels(path, filename_to_text, char_to_label):
    image_name = tf.strings.split(path, os.path.sep)[-1]
    image_text = filename_to_text.lookup(image_name)
    labels = char_to_label.lookup(split_chars(image_text))
    return labels


def split_chars(tensor, encoding="UTF-8"):
    return tf.strings.unicode_split(tensor, encoding)
