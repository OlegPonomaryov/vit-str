"""Tools for the MJSynth dataset."""
import tensorflow as tf
from pathlib import Path

import datasets


# Damaged image files that should be skipped
BAD_IMAGES = {
    "train": {
        "./2194/2/334_EFFLORESCENT_24742.jpg", "./2128/2/369_REDACTED_63458.jpg", "./2069/4/192_whittier_86389.jpg",
        "./2025/2/364_SNORTERS_72304.jpg", "./2013/2/370_refract_63890.jpg", "./1881/4/225_Marbling_46673.jpg",
        "./1863/4/223_Diligently_21672.jpg", "./1817/2/363_actuating_904.jpg", "./1730/2/361_HEREON_35880.jpg",
        "./1696/4/211_Queened_61779.jpg", "./1650/2/355_stony_74902.jpg", "./1332/4/224_TETHERED_78397.jpg",
        "./936/2/375_LOCALITIES_44992.jpg", "./913/4/231_randoms_62372.jpg", "./905/4/234_Postscripts_59142.jpg",
        "./869/4/234_TRIASSIC_80582.jpg", "./627/6/83_PATRIARCHATE_55931.jpg", "./596/2/372_Ump_81662.jpg",
        "./554/2/366_Teleconferences_77948.jpg", "./495/6/81_MIDYEAR_48332.jpg", "./429/4/208_Mainmasts_46140.jpg",
        "./384/4/220_bolts_8596.jpg", "./368/4/232_friar_30876.jpg", "./275/6/96_hackle_34465.jpg",
        "./264/2/362_FORETASTE_30276.jpg", "./173/2/358_BURROWING_10395.jpg"
    },
    "val": {
        "./2557/2/351_DOWN_23492.jpg", "./2540/4/246_SQUAMOUS_73902.jpg", "./2489/4/221_snored_72290.jpg"
    },
    "test": {
        "./2911/6/77_heretical_35885.jpg", "./2852/6/60_TOILSOME_79481.jpg", "./2749/6/101_Chided_13155.jpg"
    }
}


def load_mjsynth(path):
    """Loads the MJSynth dataset as tf.data.Dataset objects.

    :param path: Path to the dataset's main directory, that contains annotation files
    :return: Train, validation and test datasets
    """
    path = Path(path)

    rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=datasets.IMAGE_SCALE, offset=datasets.IMAGE_OFFSET)

    labels = tf.range(len(datasets.CHARS))
    char_to_label = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(datasets.CHARS, labels), -1)

    train_ds = build_dataset(path / "annotation_train.txt", rescaling, char_to_label, BAD_IMAGES["train"])
    val_ds = build_dataset(path / "annotation_val.txt", rescaling, char_to_label, BAD_IMAGES["val"])
    test_ds = build_dataset(path / "annotation_test.txt", rescaling, char_to_label, BAD_IMAGES["test"])

    return train_ds, val_ds, test_ds


def build_dataset(annotations_filename, rescaling, char_to_label, bad_images):
    paths_ds = tf.data.Dataset.from_generator(
        lambda: get_image_filenames(annotations_filename, bad_images),
        output_types=tf.string, output_shapes=[])
    return paths_ds.map(
        lambda image_path: process_path(image_path, rescaling, char_to_label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_image_filenames(annotations_filename, bad_images):
    annotations_filename = Path(annotations_filename)
    with open(annotations_filename, "r") as f:
        for line in f:
            image = line.split(" ")[0]
            if image in bad_images:
                continue
            yield (annotations_filename.parent / image).as_posix()


def process_path(path, rescaling, char_to_label):
    image = load_image(path, rescaling)
    text_labels = get_text_labels(path, char_to_label)
    return image, text_labels


def load_image(path, rescaling):
    image_data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_data, channels=datasets.IMAGE_CHANNELS)
    image = tf.image.resize(image, [datasets.IMAGE_HEIGHT, datasets.IMAGE_WIDTH])
    image = rescaling(image)
    return image


def get_text_labels(path, char_to_label):
    image_text = tf.strings.split(path, "_")[1]
    image_text_labels = char_to_label.lookup(split_chars(image_text))
    return image_text_labels


def split_chars(tensor, encoding="UTF-8"):
    return tf.strings.unicode_split(tensor, encoding)
