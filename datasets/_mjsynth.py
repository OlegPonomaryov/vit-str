import tensorflow as tf
from pathlib import Path

import datasets


def load_mjsynth(path):
    """Loads MJSynth dataset as tf.data.Dataset objects

    :param path: Path to the dataset's main directory, that contains annotation files
    :return: Train, validation and test datasets
    """
    path = Path(path)

    rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=datasets.IMAGE_SCALE, offset=datasets.IMAGE_OFFSET)

    labels = tf.range(len(datasets.CHARS))
    char_to_label = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(datasets.CHARS, labels), -1)

    train_ds = build_dataset(path / "annotation_train.txt", rescaling, char_to_label)
    val_ds = build_dataset(path / "annotation_val.txt", rescaling, char_to_label)
    test_ds = build_dataset(path / "annotation_test.txt", rescaling, char_to_label)

    return train_ds, val_ds, test_ds


def build_dataset(annotations_filename, rescaling, char_to_label):
    paths_ds = tf.data.Dataset.from_generator(
        lambda: get_image_filenames(annotations_filename),
        output_types=tf.string, output_shapes=[])
    return paths_ds.map(
        lambda image_path: process_path(image_path, rescaling, char_to_label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_image_filenames(annotations_filename):
    with open(annotations_filename, "r") as f:
        for line in f:
            yield (annotations_filename.parent / line.split(" ")[0]).as_posix()


def process_path(path, rescaling, char_to_label):
    image = load_image(path, rescaling)
    text_labels = get_text_labels(path, char_to_label)
    return image, text_labels


def load_image(path, rescaling):
    image_data = tf.io.read_file(path)
    image = tf.image.decode_png(image_data, channels=datasets.IMAGE_CHANNELS)
    image = tf.image.resize(image, [datasets.IMAGE_HEIGHT, datasets.IMAGE_WIDTH])
    image = rescaling(image)
    return image


def get_text_labels(path, char_to_label):
    image_text = tf.strings.split(path, "_")[1]
    image_text_labels = char_to_label.lookup(split_chars(image_text))
    return image_text_labels


def split_chars(tensor, encoding="UTF-8"):
    return tf.strings.unicode_split(tensor, encoding)