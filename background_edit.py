import os
import sys
import argparse
from keras.models import load_model
from scipy.misc import imresize
from skimage import filters, img_as_ubyte, io
import numpy as np
import tensorflow as tf
import cv2


model = load_model('extract_foreground_model.hdf5')

graph = tf.get_default_graph()


def predict(image):
    """
    Predicts the contours of a person on the image.

    :param image: Image on which the prediction happens
    :return: The mask with prediction
    """
    with graph.as_default():
        # Make prediction
        prediction = model.predict(image[None, :, :, :])

    prediction = prediction.reshape((224, 224, -1))
    return prediction


def overlay_transparent(background, overlay):
    """
    Overlays two images over each other. It assumes that both pictures have the
    same size. Picture to be overlayed should have transparency channel.

    :param background: background picture
    :param overlay: picture to be put on top of the background
    :return: merged picture
    """
    h, w = overlay.shape[0], overlay.shape[1]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [overlay, np.ones((h, w, 1), dtype=overlay.dtype) * 255],
            axis=2, )

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[:h, :w] = (1.0 - mask) * background[:h, :w] + mask * overlay_img

    return background


def reduce_background(img, saturation, brightness_factor):
    """
    Reduces the saturation of the image and makes it brighter.

    :param img: image to be processed
    :param saturation: number by which the saturation should be reduced
    :param brightness_factor: factor by which the brightness shall be changed
    :return: modified picture
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v += np.uint8((255 - v) / brightness_factor)
    s[s < saturation] = 0
    s[s >= saturation] -= saturation

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def process_background(image):
    """
    Smooths received image and calls the function to make it brighter and less
    saturated.

    :param image: image to be processed
    :return: processed image
    """
    background = np.array(image)
    smoothed = filters.gaussian(
        background, sigma=80, multichannel=True, mode='reflect')
    reduced_background = reduce_background(img_as_ubyte(smoothed), 120, 1.5)
    return reduced_background


def retrieve_person(image):
    """
    Uses neural network to find a person on a photo.

    :param image: image to retrieve the person
    :return: retrieved person (numpy array with transparency set to the areas
            that person was not discovered)
    """
    img = image.copy()
    head = np.array(cv2.resize(img, (224, 224))) / 255.0
    head_prediction_small = predict(head[:, :, 0:3])
    head_prediction = imresize(
        head_prediction_small[:, :, 1], (image.shape[0], image.shape[1]))
    # prediction almost always returns ~1/40 lowest rows as not being part of
    # person, but it is almost always part of it. Correct it.
    to_correct = int(image.shape[0]/40)
    head_prediction[image.shape[0] - to_correct:, :] = \
        head_prediction[image.shape[0] - to_correct:
                        image.shape[0] - (to_correct - 1), :]
    person = np.append(img, head_prediction[:, :, None], axis=-1)
    return person


def process_img_background(image, name, location):
    """
    Calls method for finding person on the image and one for processing the
    background. After all it saves the modified picture.

    :param image: image to be processed
    :param name: name under which the image shall be saved
    :param location: location where the image shall be saved
    """
    face = retrieve_person(image)
    background = process_background(image)
    joined_image = overlay_transparent(background, face)
    final_img = cv2.cvtColor(joined_image, cv2.COLOR_BGR2RGB)
    file_path = os.path.join(location, name + '.png')
    cv2.imwrite(file_path, final_img)


def cli_argument_parser(argv):
    """
    Argument parser.

    :param argv: commandline arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Soften background.')
    parser.add_argument('--photos_folder',
                        help='folder containing photos to edit',
                        default='sample')
    parser.add_argument('--photos_format',
                        help='format of photos to be edited, i.e.: jpg',
                        default='jpg')
    parser.add_argument('--output_folder',
                        help='folder where to store result',
                        default='out')
    return parser.parse_args(argv)


def main(argv):
    p = cli_argument_parser(argv)
    if not os.path.exists(p.photos_folder):
        raise FileExistsError("Provided photos folder does not exist.")
    if not os.path.exists(p.output_folder):
        os.makedirs(p.output_folder)
    ic = io.ImageCollection(p.photos_folder + '/*.' + p.photos_format)
    for i in range(len(ic)):
        try:
            print("processing ", os.path.basename(ic.files[i][:-4]))
            process_img_background(
                ic[i], os.path.basename(ic.files[i][:-4]), p.output_folder)
        except Exception as e:
            print("Exception happened!", e)


if __name__ == '__main__':
    main(sys.argv[1:])
