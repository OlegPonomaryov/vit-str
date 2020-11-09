"""Tools for notebooks"""
import matplotlib
import matplotlib.pyplot as plt


def plot_images(images, text, rows, columns):
    """Plots images with text.

    :param images: A collection of images
    :param text: Text for each image
    :param rows: Number of subplot rows
    :param columns: Number of subplot columns
    """
    if len(images) != len(text):
        raise ValueError("Images and text collections should have the same length")

    if len(images) > rows * columns:
        raise ValueError("The number of images should be less than or equal to the number of subplots")

    for i in range(len(images)):
        plt.subplot(rows, columns, i + 1)
        plt.title(text[i], {'fontsize': 18})
        plt.imshow(images[i], cmap="binary", norm=matplotlib.colors.Normalize(-1, 1))
