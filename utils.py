import tensorflow as tf
from keras import backend as keras

# functions for preparation of dataset

def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "masks")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return (image,  mask)

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def resize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.image.resize(input_image,(128,128))
    input_mask = tf.image.resize(input_mask,(128,128))
    return input_image, input_mask

# dice score function

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
    return dice