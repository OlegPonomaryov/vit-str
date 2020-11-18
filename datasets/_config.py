"""Constants used throughout the package."""
import string


# A list of all unique characters that the model is trained to recognize
CHARS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# Dimensions of images used for training and testing
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 1

# Parameters for image rescaling
IMAGE_SCALE = 1./127.5
IMAGE_OFFSET = -1
