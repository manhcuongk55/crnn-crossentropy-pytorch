import os

import torch
import torch.utils.data
import torch.utils.data
from constant import MAX_LEN

# disable warnings about AVX CPU in tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable other commons warnings in console
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, dtype=torch.long)
    return y[labels]


def one_hot_digits(digits):
    cleaned_digits = []
    for digit in digits:
        digit = digit.replace('.', '10')
        digit = digit.replace('-', '11')
        cleaned_digits.append(int(digit))

    cleaned_digits.append(12)
    cleaned_digits += [0] * (MAX_LEN - len(cleaned_digits))

    return one_hot_embedding(cleaned_digits, 13)
