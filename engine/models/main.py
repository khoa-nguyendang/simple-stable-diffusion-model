import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)

